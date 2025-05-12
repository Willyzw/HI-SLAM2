import torch
import lietorch
import droid_backends

from lietorch import SE3, Sim3
from torch.multiprocessing import Value

from modules.droid_net import cvx_upsample
import geom.projective_ops as pops
from geom.ba import JDSA
from pgo_buffer import global_relative_posesim3_constraints

class DepthVideo:
    def __init__(self, config, image_size, buffer):

        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.is_initialized = False
        self.config = config

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.poses_sim3 = torch.zeros(buffer, 8, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.disps_prior = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_prior_up = torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()
        self.normals = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.float)
        self.semantics = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.uint8)

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        self.poses_sim3[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float, device="cuda")

        # depth prior scale
        self.dscales = torch.ones(buffer, 2, 2, device='cuda', dtype=torch.float).share_memory_()
        self.doffset = torch.zeros(buffer, 1, 1, device='cuda', dtype=torch.float).share_memory_()

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            self.disps_prior_up[index] = 1.0/item[4]
            depth = item[4][3::8,3::8]
            self.disps_prior[index] = torch.where(depth>0, 1.0/depth, 0).cuda()

        if item[5] is not None:
            self.normals[index] = item[5]

        if item[6] is not None:
            self.intrinsics[index] = item[6]
        else:
            self.intrinsics[index] = self.intrinsics[0].clone()

        if len(item) > 7:
            self.fmaps[index] = item[7]

        if len(item) > 8:
            self.nets[index] = item[8]

        if len(item) > 9:
            self.inps[index] = item[9]
        
        if len(item) > 10:
            self.semantics[index] = item[10]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def shift(self, ix, n=1):
        with self.get_lock():
            self.tstamp[ix+n:self.counter.value+n] = self.tstamp[ix:self.counter.value].clone()
            self.images[ix+n:self.counter.value+n] = self.images[ix:self.counter.value].clone()
            self.semantics[ix+n:self.counter.value+n] = self.semantics[ix:self.counter.value].clone()
            self.dirty[ix+n:self.counter.value+n] = self.dirty[ix:self.counter.value].clone()
            self.poses[ix+n:self.counter.value+n] = self.poses[ix:self.counter.value].clone()
            self.poses_sim3[ix+n:self.counter.value+n] = self.poses_sim3[ix:self.counter.value].clone()
            self.disps[ix+n:self.counter.value+n] = self.disps[ix:self.counter.value].clone()
            self.disps_prior[ix+n:self.counter.value+n] = self.disps_prior[ix:self.counter.value].clone()
            self.disps_up[ix+n:self.counter.value+n] = self.disps_up[ix:self.counter.value].clone()
            self.disps_prior_up[ix+n:self.counter.value+n] = self.disps_prior_up[ix:self.counter.value].clone()
            self.intrinsics[ix+n:self.counter.value+n] = self.intrinsics[ix:self.counter.value].clone()
            self.normals[ix+n:self.counter.value+n] = self.normals[ix:self.counter.value].clone()
            self.fmaps[ix+n:self.counter.value+n] = self.fmaps[ix:self.counter.value].clone()
            self.nets[ix+n:self.counter.value+n] = self.nets[ix:self.counter.value].clone()
            self.inps[ix+n:self.counter.value+n] = self.inps[ix:self.counter.value].clone()
            self.counter.value += n

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze().cpu()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean().item() * self.config['Dataset']['scale_multiplier']
            self.poses[:self.counter.value,:3] *= s
            self.disps[:self.counter.value] /= s
            self.disps_up[:self.counter.value] /= s
            self.dscales[:self.counter.value] /= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj, sim3=False):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = Sim3(self.poses_sim3[None]) if sim3 else SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
        
    def distance_covis(self, ii=None):
        """ frame distance metric based on covisibility """
        ii = torch.as_tensor(ii)
        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        poses = self.poses[:self.counter.value].clone()
        d = droid_backends.covis_distance(poses, self.disps, self.intrinsics[0], ii)
        d = d * (1. / self.disps[ii].median())
        return d

    def cuda_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, use_mono=False):
        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            ht, wd = self.disps.shape[1:]
            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            if use_mono:
                poses = lietorch.SE3(self.poses[:t1][None])
                disps = self.disps[:t1][None]
                dscales = self.dscales[:t1]
                disps, dscales, _ = JDSA(target, weight, eta, poses, disps, self.intrinsics[None], self.disps_prior, dscales, ii, jj, self.mono_depth_alpha)
                self.disps[:t1] = disps[0]
                self.dscales[:t1] = dscales

            self.disps.clamp_(min=0.001)

    def cuda_pgba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, verbose=False):
        from geom.ba import pose_retr

        poses = Sim3(self.poses_sim3[:t1][None])

        # rel pose constraints
        rel_N = self.pgobuf.rel_N.value
        iip, jjp = self.pgobuf.rel_ii[:rel_N].cuda(), self.pgobuf.rel_jj[:rel_N].cuda()
        rel_poses = self.pgobuf.rel_poses[:rel_N].cuda()[None]
        infos = 1 / self.pgobuf.rel_covs[:rel_N].cuda()
        infos = torch.cat((infos, infos.min(dim=1, keepdim=True)[0]), dim=1)
        infos = infos.unsqueeze(2).expand(*infos.size(), infos.shape[-1]) * torch.eye(infos.shape[-1], device='cuda')[None]
        infos[torch.isnan(infos) | torch.isinf(infos)] = 0.

        for _ in range(itrs):
            Hsp, vsp, pchi2, pchi2_scaled = global_relative_posesim3_constraints(iip, jjp, poses, rel_poses, infos, pw=1e-3)

            ht, wd = self.disps.shape[1:]
            dx, dz = droid_backends.pgba(poses.data[0], self.disps, self.intrinsics[0], 
                                target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous(),
                                weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous(), eta,
                                Hsp, vsp, ii, jj, iip, jjp, t0, t1, lm, ep)
            # print(dx.mean(), dz.mean())
            poses = pose_retr(poses, dx[None], torch.arange(t0, t1))

        self.poses_sim3[:t1] = poses.data
        self.disps.clamp_(min=0.001, max=10)
