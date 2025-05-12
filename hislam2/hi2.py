import os
import torch
import numpy as np
from lietorch import SE3

from modules.droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from track_frontend import TrackFrontend
from track_backend import TrackBackend
from util.trajectory_filler import PoseTrajectoryFiller
from util.utils import load_config

from collections import OrderedDict
from torch.multiprocessing import Process, Queue
from gs_backend import GSBackEnd
from pgo_buffer import PGOBuffer


class Hi2:
    def __init__(self, args):
        super(Hi2, self).__init__()
        self.load_weights(args.weights)
        self.config = config = load_config(args.config)
        self.args = args
        self.images = {}
        self.semantic_images = {}

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(config, args.image_size, args.buffer)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, config["Tracking"]["motion_filter"])

        # frontend process
        self.frontend = TrackFrontend(self.net, self.video, config["Tracking"]["frontend"])

        # backend process
        self.backend = TrackBackend(self.net, self.video, config["Tracking"]["backend"])

        # 3dgs
        self.gs = GSBackEnd(config, self.args.output, args.gsvis)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        # visualizer
        if args.droidvis:
            from util.droid_visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # global PGBA backend
        self.pgba = config["Tracking"]["pgba"]["active"]
        if self.pgba:
            self.video.pgobuf = PGOBuffer(self.net, self.video, self.frontend, config["Tracking"]["pgba"])
            self.LC_data_queue = Queue()
            self.video.pgobuf.set_LC_data_queue(self.LC_data_queue)
            self.mp_backend = Process(target=self.video.pgobuf.spin)
            self.mp_backend.start()

    def load_weights(self, weights):
        """ load trained model weights """
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def call_gs(self, viz_idx, dposes=None, dscale=None):
        data = {'viz_idx':  viz_idx.to(device='cpu'),
                'tstamp':   self.video.tstamp[viz_idx].to(device='cpu'),
                'poses':    self.video.poses[viz_idx].to(device='cpu'),
                'images':   self.video.images[viz_idx.cpu()],
                'normals':  self.video.normals[viz_idx.cpu()],
                'depths':   1./self.video.disps_up[viz_idx.cpu()],
                'semantics': self.video.semantics[viz_idx.cpu()],
                'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu') * 8,
                'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
                'scale_updates': dscale.to(device='cpu') if dscale is not None else None}
        self.gs.process_track_data(data)

    def track(self, tstamp, image, semantic=None, intrinsics=None, is_last=False):
        """ main thread - update map """

        with torch.no_grad():
            self.images[tstamp] = image
            self.semantic_images[tstamp] = semantic

            # check there is enough motion
            self.filterx.track(tstamp, image, semantic, intrinsics, is_last)

            # local bundle adjustment
            viz_idx = self.frontend(is_last=is_last)

        if len(viz_idx) and self.pgba:
            dposes, dscale = self.video.pgobuf.run_pgba(self.LC_data_queue)
            if dposes is not None:
                self.call_gs(torch.arange(0, self.video.counter.value-1, device='cuda'), dposes[:-1], dscale[:-1])

        if len(viz_idx):
            self.call_gs(viz_idx)

    def terminate(self):
        """ terminate the visualization process, return poses [t, q] """
        self.video.ready.value = 1
        if self.pgba:
            dposes, dscale = self.video.pgobuf.run_pgba(self.LC_data_queue)
            if dposes is not None:
                self.call_gs(torch.arange(0, self.video.counter.value, device='cuda'), dposes, dscale)
            self.mp_backend.terminate()
        del self.frontend

        # check if new keyframes need to be added
        deltas = np.add.accumulate(self.filterx.deltas)
        d_covis = self.video.distance_covis(torch.arange(1, self.video.counter.value-1, device='cuda'))
        new_kfs = []
        for i in torch.arange(1, self.video.counter.value-1, device='cuda'):
            if d_covis[i-1] > self.config['Tracking']['backend']['covis_thresh']:
                delta = deltas[int(self.video.tstamp[i-1])] + (deltas[int(self.video.tstamp[i])] - deltas[int(self.video.tstamp[i-1])]) / 2
                ind1 = np.where(deltas > delta)[0][0]
                if ind1 not in self.video.tstamp:
                    new_kfs.append(ind1)
                delta = deltas[int(self.video.tstamp[i])] + (deltas[int(self.video.tstamp[i+1])] - deltas[int(self.video.tstamp[i])]) / 2
                ind2 = np.where(deltas > delta)[0][0]
                if ind2 not in self.video.tstamp:
                    new_kfs.append(ind2)
                print(f' - add new keyframe {ind1} and {ind2} for {self.video.tstamp[i].item()}')
        new_kfs = sorted(list(set(new_kfs)))

        # fill in poses for new keyframes
        for i in range(0, len(new_kfs), 10):
            new_kf = new_kfs[i:i+10]
            images = [self.images[i] for i in new_kf]
            Gs, gmap = self.traj_filler.fill(new_kf, images, return_fmap=True)
            inputs = torch.stack(images).cuda() / 255.0
            inputs = inputs.sub_(self.filterx.MEAN).div_(self.filterx.STDV)
            net, inp = self.filterx.context_encoder(inputs[:,[0]])
            for i, ind in enumerate(new_kf):
                place = (self.video.tstamp > ind).nonzero()[0].item()
                self.video.shift(place)
                depth, normal = self.filterx.prior_extractor(inputs[i])
                self.video[place] = (ind, images[i], Gs.data[i], self.video.disps[place].mean(), depth.cpu(), normal.cpu(), None, gmap[i], net[i,0], inp[i,0])
        del self.filterx

        # global bundle adjustment
        poses_pre = self.video.poses[:self.video.counter.value].clone()
        self.backend(4)
        self.backend(8)
        del self.backend
        poses_pos = self.video.poses[:self.video.counter.value].clone()
        dposes = SE3(poses_pos).inv() * SE3(poses_pre)
        dscale = torch.ones(self.video.counter.value, 1)
        torch.cuda.empty_cache()

        # final refinement
        self.call_gs(torch.arange(0, self.video.counter.value, device='cuda'), dposes, dscale)
        updated_poses = self.gs.finalize()
        self.video.poses[:self.video.counter.value] = torch.tensor(updated_poses[:,1:])

        traj_full = self.traj_filler(self.images)
        self.gs.eval_rendering(self.images, self.semantic_images, self.args.gtdepthdir, traj_full.matrix().data, self.video.tstamp[:self.video.counter.value].to(device='cpu'))
        return traj_full.inv().data.cpu().numpy()
