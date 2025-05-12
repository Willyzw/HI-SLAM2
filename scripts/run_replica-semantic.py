import os
import ast
import numpy as np
import open3d as o3d
from collections import defaultdict
from glob import glob
from shutil import copyfile


out = 'outputs/replica-semantic'
os.makedirs(f'{out}/meshes', exist_ok=True)
seqs = sorted(glob('data/Replica-semantic/ro*')) + sorted(glob('data/Replica-semantic/off*'))

metrics = defaultdict(float)
for seq in seqs:
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)

    # run HI-SLAM2
    cmd = f'python demo.py --imagedir {seq}/frames --semanticdir {seq}/semantic_colors --gtdepthdir {seq}/depths '
    cmd += f'--config config/replica_config.yaml --calib calib/replica.txt --output {out}/{name} > {out}/{name}/log.txt'
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        os.system(cmd)

    # eval ate
    if not os.path.exists(f'{out}/{name}/ape.txt') or len(open(f'{out}/{name}/ape.txt').readlines()) < 10:
        os.system(f'evo_ape tum {seq}/traj_tum.txt {out}/{name}/traj_full.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
        os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
    ATE = float([i for i in open(f'{out}/{name}/ape.txt').readlines() if 'rmse' in i][0][-10:-1]) * 100
    metrics['ATE'] += ATE

    # eval render
    psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
    metrics['PSNR'] += psnr['mean_psnr']
    metrics['SSIM'] += psnr['mean_ssim']
    metrics['LPIPS'] += psnr['mean_lpips']
    metrics['MIoU'] += psnr['mean_miou'] * 100

    # run tsdf fusion
    w = 2
    weight = f'w{w:.1f}'
    if not os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}.ply'):
        os.system(f'python tsdf_integrate.py --result {out}/{name} --voxel_size 0.006 --weight {w} > /dev/null')
        ply = o3d.io.read_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}.ply')
        ply = ply.transform(np.load(f'{out}/{name}/evo/alignment_transformation_sim3.npy'))
        o3d.io.write_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', ply)
        copyfile(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', f'{out}/meshes/{name}_semantic.ply')
    
    # eval 3d recon
    if not os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
        os.system(f'python scripts/eval_recon.py {out}/{name}/tsdf_mesh_{weight}_aligned.ply data/Replica/gt_mesh_culled/{name}.ply --eval_3d --save {out}/{name}/eval_recon_{weight}.txt > /dev/null')
    result = ast.literal_eval(open(f'{out}/{name}/eval_recon_{weight}.txt').read())
    metrics['accu'] += result['mean precision']
    metrics['comp'] += result['mean recall']
    metrics['compr'] += result['recall']
    print(f"{name} - ATE: {ATE:.4f}, psnr: {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}, miou: {psnr['mean_miou']*100:.2f}, acc: {result['mean precision']:.3f}, comp: {result['mean recall']:.3f}, comp rat: {result['recall']:.3f}")

avg = ""
for r in metrics:
    avg += f'{r}: {metrics[r]/len(seqs):.4f} \t'
print(f'- avg: {avg}')
