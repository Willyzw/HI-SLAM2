import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2


def show_image(image, depth_prior, depth, normal, semantic):
    from util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()
    semantic = semantic[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    cv2.imshow('rgb / prior normal / aligned prior depth / JDSA depth', np.concatenate((image / 255.0, (normal[...,[2,1,0]]+1.)/2., depth, semantic / 255.0), axis=1)[::2,::2])
    cv2.waitKey(1)


def mono_stream(queue, imagedir, semanticdir, calib, undistort=False, cropborder=False, start=0, length=100000):
    """ image generator """
    RES = 341 * 640

    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])

    image_list = sorted(os.listdir(imagedir))[start:start+length]
    image_list = [x for x in image_list if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')]
    if semanticdir is not None:
        semantic_list = sorted(os.listdir(semanticdir))[start:start+length]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if semanticdir is not None:
            semantic = cv2.imread(os.path.join(semanticdir, semantic_list[t]))
            semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        intrinsics = torch.tensor(calib[:4])
        # if len(calib) > 4 and undistort:
        #     image = cv2.undistort(image, K, calib[4:])
        # if cropborder > 0:
        #     image = image[cropborder:-cropborder, cropborder:-cropborder]
        #     intrinsics[2:] -= cropborder

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        image = cv2.resize(image, (w1, h1))
        image = torch.as_tensor(image).permute(2, 0, 1)
        if semanticdir is not None:
            semantic = cv2.resize(semantic, (w1, h1), interpolation=cv2.INTER_NEAREST)
            semantic = torch.as_tensor(semantic).permute(2, 0, 1)

        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)

        is_last = (t == len(image_list)-1)
        queue.put((t, image[None], semantic, intrinsics[None], is_last))

    time.sleep(10)


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(imagedir))[start:]])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)                     #  for evo evaluation 
    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--semanticdir", type=str, help="path to semantic directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')

    hi2 = None
    queue = Queue(maxsize=8)
    reader = Process(target=mono_stream, args=(queue, args.imagedir, args.semanticdir, args.calib, args.undistort, args.cropborder, args.start, args.length))
    reader.start()

    N = len(os.listdir(args.imagedir))
    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
    pbar = tqdm(range(N), desc="Processing keyframes")
    while 1:
        (t, image, semantic, intrinsics, is_last) = queue.get()
        pbar.update()

        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)

        hi2.track(t, image, semantic, intrinsics=intrinsics, is_last=is_last)

        if args.droidvis and hi2.video.tstamp[hi2.video.counter.value-1] == t:
            from geom.ba import get_prior_depth_aligned
            index = hi2.video.counter.value-2
            depth_prior, _ = get_prior_depth_aligned(hi2.video.disps_prior_up[index][None].cuda(), hi2.video.dscales[index][None])
            show_image(image[0], 1./depth_prior.squeeze(), 1./hi2.video.disps_up[index], hi2.video.normals[index], hi2.video.semantics[index])
        pbar.set_description(f"Processing keyframe {hi2.video.counter.value} gs {hi2.gs.gaussians._xyz.shape[0]}")

        if is_last:
            pbar.close()
            break

    reader.join()

    traj = hi2.terminate()
    save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)

    print("Done")
