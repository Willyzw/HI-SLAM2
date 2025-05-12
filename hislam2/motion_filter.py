import cv2
import torch
import lietorch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import geom.projective_ops as pops

from modules.corr import CorrBlock
from torchvision import transforms
from midas.omnidata import OmnidataModel


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, config, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = config["thresh"]
        self.init_thresh = config["init_thresh"] if "init_thresh" in config else self.thresh
        self.device = device

        self.count = 0
        self.omni_dep = None
        self.deltas = [0]

        self.skip_blur = config["skip_blur"]
        self.cache = [None]*5
        self.shapeness = [0]*5

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def prior_extractor(self, im_tensor):
        input_size = im_tensor.shape[-2:]
        trans_totensor = transforms.Compose([transforms.Resize((512, 512), antialias=True)])
        im_tensor = trans_totensor(im_tensor).cuda()
        if self.omni_dep is None:
            self.omni_dep = OmnidataModel('depth', 'pretrained_models/omnidata_dpt_depth_v2.ckpt', device="cuda:0")
            self.omni_normal = OmnidataModel('normal', 'pretrained_models/omnidata_dpt_normal_v2.ckpt', device="cuda:0")
        depth = self.omni_dep(im_tensor)[None] * 50
        depth = F.interpolate(depth, input_size, mode='bicubic')
        depth = depth.float().squeeze()
        normal = self.omni_normal(im_tensor) * 2.0 - 1.0
        normal = F.interpolate(normal, input_size, mode='bicubic')
        normal = normal.float().squeeze()
        return depth, normal

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, semantic=None, intrinsics=None, is_last=False):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8
        intrinsics[:,:4] /= 8.0

        s = sharpness(image[0].permute(1,2,0).cpu().numpy())

        # normalize images
        inputs = image[None].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            depth, normal = self.prior_extractor(inputs[0])
            net, inp = self.context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], None, 1.0, depth, normal, intrinsics, gmap, net[0], inp[0], semantic)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            self.deltas.append(delta.norm(dim=-1).mean().item())
            # check motion magnitue / add new frame to video
            thresh = self.init_thresh if not self.video.is_initialized else self.thresh
            if delta.norm(dim=-1).mean().item() > thresh or is_last:
                index_min = np.argmax(self.shapeness)
                if self.skip_blur and self.shapeness[index_min] > s:
                    tstamp, image, intrinsics, gmap, inputs = self.cache[index_min]
                self.shapeness = [0]*5
                self.cache = [None]*5

                depth, normal = self.prior_extractor(inputs[0])
                self.count = 0
                net, inp = self.context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, normal, intrinsics, gmap, net[0], inp[0], semantic)

            else:
                self.shapeness[tstamp%5] = s
                self.cache[tstamp%5] = [tstamp, image, intrinsics, gmap, inputs]
                self.count += 1
