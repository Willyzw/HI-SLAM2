import torch
import time
import numpy as np
import geom.projective_ops as pops

from torch.multiprocessing import Value
from queue import Empty
from scipy.spatial.transform import Rotation as R
from lietorch import SE3, Sim3
from factor_graph import FactorGraph
from util.utils import Log


eps = 1e-8
def diff(x1, x2):
    return (x1 - x2) / 2. / eps


def num_jacobi(func, Gi, Gj=None, first=True):
    batch = Gi.shape[0]
    D = Gi.manifold_dim
    J_num = []
    for i in range(D):
        delta = torch.zeros(D, device='cuda').expand(batch, 1, D).type(torch.float64)
        delta[:, :, i] = eps
        if first:
            if isinstance(Gi, Sim3):
                J_num.append(diff(func(Sim3.exp(delta)*Gi, Gj), func(Sim3.exp(-delta)*Gi, Gj)))
            else:
                J_num.append(diff(func(SE3.exp(delta)*Gi, Gj), func(SE3.exp(-delta)*Gi, Gj)))
        else:
            if isinstance(Gi, Sim3):
                J_num.append(diff(func(Gi, Sim3.exp(delta)*Gj), func(Gi, Sim3.exp(-delta)*Gj)))
            else:
                J_num.append(diff(func(Gi, SE3.exp(delta)*Gj), func(Gi, SE3.exp(-delta)*Gj)))
    return torch.stack(J_num, dim=-1)


def global_relative_posesim3_constraints(ii, jj, poses, rel_poses, infos, pw=1e-5, verbose=False):
    torch.set_printoptions(precision=8, sci_mode=False, linewidth=120)

    Gii = Sim3((poses[:, ii].data.type(torch.float64)))
    Gjj = Sim3((poses[:, jj].data.type(torch.float64)))
    Gij = Sim3(SE3(rel_poses.data.type(torch.float64)))

    def func(Gii, Gjj):
        e = Gij * Gii * Gjj.inv()
        return e.log()

    # numerical jacobi
    Ji = num_jacobi(func, Gii, Gjj, first=True).type(torch.float32)
    Jj = num_jacobi(func, Gii, Gjj, first=False).type(torch.float32)

    r = func(Gii, Gjj).unsqueeze(-1).type(torch.float32)
    chi2 = torch.sum(r.transpose(2, 3) @ r)
    chi2_scaled = torch.sum(r.transpose(2, 3) @ infos @ r)

    wJiT = ((pw * Ji.double()).transpose(2, 3) @ infos.double()).float()
    wJjT = ((pw * Jj.double()).transpose(2, 3) @ infos.double()).float()
    Hsp = torch.stack([torch.matmul(wJiT, Ji), torch.matmul(wJiT, Jj), torch.matmul(wJjT, Ji), torch.matmul(wJjT, Jj)])    # 4x1xNx7x7
    vsp = -torch.stack([torch.matmul(wJiT, r), torch.matmul(wJjT, r)]).squeeze(-1)  # 2x1xNx7
    return Hsp, vsp, chi2, chi2_scaled


class PGOBuffer:
    def __init__(self, net, video, frontend, config):
        self.net = net
        self.video = video
        self.frontend = frontend
        max_rel = 2e5
        self.pgba_thresh = config["pgba_thresh"]

        self.rel_N = Value('i', 0)
        self.rel_ii = torch.zeros(int(max_rel), dtype=torch.long, device='cpu').share_memory_()
        self.rel_jj = torch.zeros(int(max_rel), dtype=torch.long, device='cpu').share_memory_()
        self.rel_poses = torch.zeros(int(max_rel), 7, device="cpu", dtype=torch.float).share_memory_()
        self.rel_covs = torch.zeros(int(max_rel), 6, device="cpu", dtype=torch.float).share_memory_()
        self.rel_valid_percent = torch.zeros(int(max_rel), device="cpu", dtype=torch.float).share_memory_()

        self.kfs = set()
        self.lcii = torch.as_tensor([], dtype=torch.long, device='cuda')
        self.lcjj = torch.as_tensor([], dtype=torch.long, device='cuda')

    @torch.cuda.amp.autocast(enabled=False)
    def add_rel_poses(self, ii, jj, target, weight):
        valid = (weight[0] > 0.1).float().mean([-3, -2, -1])
        masks = valid > 1e-2
        ii, jj, target, weight = ii[masks], jj[masks], target[:, masks], weight[:, masks]

        N = ii.shape[0]
        if N < 1:
            return

        t1 = max(ii.max(), jj.max())+1
        poses = SE3(self.video.poses[:t1][None])
        rel_poses = poses[:, jj] * poses[:, ii].inv()

        for _ in range(4):
            coords, valid, (_, Jj, _) = pops.projective_transform(
                poses, self.video.disps[None], self.video.intrinsics[None], ii, jj, jacobian=True)
            r = (target - coords).view(1, N, -1, 1)
            w = .001 * (valid * weight).view(1, N, -1, 1)
            Jj = Jj.reshape(1, N, -1, 6)
            wJjT = (w * Jj).transpose(2, 3)
            Hjj = torch.matmul(wJjT, Jj) + 1e-4*torch.eye(6, device='cuda')[None, None]
            vj = torch.matmul(wJjT, r)

            Hinv = torch.linalg.inv(Hjj)
            dx = Hinv @ vj
            rel_poses = rel_poses.retr(dx.squeeze(-1))

        V = Jj @ dx - r
        sig2 = (w * V).transpose(2, 3) @ V
        cov = sig2 * Hinv
        cov = torch.diagonal(cov, dim1=2, dim2=3)

        valid = (weight[0] > 0.1).float().mean([-3, -2, -1])
        self.rel_ii[self.rel_N.value:self.rel_N.value+N] = ii
        self.rel_jj[self.rel_N.value:self.rel_N.value+N] = jj
        self.rel_poses[self.rel_N.value:self.rel_N.value+N] = rel_poses.data[0]
        self.rel_covs[self.rel_N.value:self.rel_N.value+N] = cov[0]
        self.rel_valid_percent[self.rel_N.value:self.rel_N.value+N] = valid
        self.rel_N.value += N

    def _pgba(self, LC_data):
        start_time = time.time()
        lcii, lcjj = LC_data['lcii'], LC_data['lcjj']
        graph = FactorGraph(self.video, self.net.update, corr_impl="volume", max_factors=-1)
        ii, jj = torch.cat((lcii, lcjj, self.frontend.graph.ii)), torch.cat((lcjj, lcii, self.frontend.graph.jj))
        graph.add_factors(ii, jj)

        t0 = max(8, min(ii.min().item(), jj.min().item())+1)
        
        kx = torch.unique(ii)
        m = sum(self.frontend.graph.ii_inac == k for k in kx).bool()
        graph.ii_inac = self.frontend.graph.ii_inac[m]
        graph.jj_inac = self.frontend.graph.jj_inac[m]
        graph.target_inac = self.frontend.graph.target_inac[:,m]
        graph.weight_inac = self.frontend.graph.weight_inac[:,m]

        with torch.no_grad():
            with self.video.get_lock():
                t1 = self.video.counter.value
                self.video.poses_sim3[:t1, 7] = 1
                self.video.poses_sim3[:t1, :7] = self.video.poses[:t1]

                graph.update_pgba(t0=t0, t1=t1)

                for _ in range(6):
                    self.frontend.graph.update(None, None, use_inactive=True)

                self.video.dirty[:self.video.counter.value] = True

        Log(f"run with {graph.ii.shape[0]} factors from keyframe {t0} to {t1} took {time.time() - start_time:.2f} seconds", tag="PGBA")

        self.add_rel_poses(ii[:2*len(lcii)], jj[:2*len(lcii)], graph.target[:,:2*len(lcii)], graph.weight[:,:2*len(lcii)])
        del graph

    def run_pgba(self, LC_data_queue):
        try:
            LC_data = LC_data_queue.get(timeout=0.001)
        except Empty:
            LC_data = None

        if LC_data:
            poses_pre = self.video.poses[:self.video.counter.value].clone()
            self._pgba(LC_data)
            poses_pos = self.video.poses[:self.video.counter.value].clone()
            dposes = SE3(poses_pos).inv() * SE3(poses_pre)
            dscale = self.video.poses_sim3[:self.video.counter.value, -1:]
            return dposes, dscale
        else:
            return None, None

    def set_LC_data_queue(self, queue):
        self.LC_data_queue = queue

    def reset(self):
        self.lcii = torch.as_tensor([], dtype=torch.long, device='cuda')
        self.lcjj = torch.as_tensor([], dtype=torch.long, device='cuda')

    def search_lc_candidate(self, hist, kx):
        ii = torch.arange(0, hist, device='cuda')
        jj = torch.ones_like(ii) * kx

        dd = self.video.distance(ii, jj)
        ls = dd < self.pgba_thresh
        if torch.sum(ls) > 0:
            ii, jj, dd = ii[ls==True], jj[ls==True], dd[ls==True]
            Gij = (SE3(self.video.poses[jj]) * SE3(self.video.poses[ii]).inv()).data
            euls = R.from_quat(Gij[:, 3:].cpu().numpy()).as_euler('zxy', degrees=True)
            oris = np.linalg.norm(euls, axis=1)

            ls = oris < 120
            if np.sum(ls) > 0:
                self.lcii = torch.cat([self.lcii, ii[ls==True][:10]])
                self.lcjj = torch.cat([self.lcjj, jj[ls==True][:10]])

    @torch.no_grad()
    def spin(self):
        while self.video.ready.value == 0:
            kx = self.video.counter.value - 4
            if kx < 60 or kx in self.kfs:
                time.sleep(0.1)
                continue

            self.search_lc_candidate(self.video.counter.value - 55, kx)
            self.kfs.add(kx)

            wait_long = len(self.lcjj) > 0 and (kx - self.lcjj[0]) > 3
            if self.lcii.shape[0] > 24 or wait_long:
                self.LC_data_queue.put({'lcii': self.lcii, 'lcjj': self.lcjj})
                self.reset()
                self.kfs.update({kx+1, kx+2})

        if self.lcii.shape[0] > 0:
            self.LC_data_queue.put({'lcii': self.lcii, 'lcjj': self.lcjj})
