#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

// #include "utils.cuh"

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;



#define MIN_DEPTH 0.25
#define CV_PI 3.1415926

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}

__device__ float2 proj(const float *Xj, const float *intrinsics)
{
  float2 xnyn;
  xnyn.x = intrinsics[0] * (Xj[0] / Xj[2]) + intrinsics[2];
  xnyn.y = intrinsics[1] * (Xj[1] / Xj[2]) + intrinsics[3];
  return xnyn;
}

__device__ void iproj(float u, float v, const float *intrinsics, float * X, float di)
{
  X[0] = (u - intrinsics[2]) / intrinsics[0];
  X[1] = (v - intrinsics[3]) / intrinsics[1];
  X[2] = 1;
  X[3] = di;
}

__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
actSim3(const float *t, const float *q, const float s, const float *X, float *Y) {
    actSO3(q, X, Y);    
    Y[3] = X[3];
    Y[0] = s * Y[0] + X[3] * t[0];
    Y[1] = s * Y[1] + X[3] * t[1];
    Y[2] = s * Y[2] + X[3] * t[2];
}

__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void
adjSim3(const float *t, const float *q, const float s, const float *X, float *Y) {
    float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
    actSO3(qinv, &X[0], &Y[0]);
    actSO3(qinv, &X[3], &Y[3]);

    float u[3], v[3];
    u[0] = t[2]*X[1] - t[1]*X[2];
    u[1] = t[0]*X[2] - t[2]*X[0];
    u[2] = t[1]*X[0] - t[0]*X[1];
    
    actSO3(qinv, u, v);
    Y[3] += v[0];
    Y[4] += v[1];
    Y[5] += v[2];

    Y[0] *= s;
    Y[1] *= s;
    Y[2] *= s;
    
    Y[3] = s * Y[3] + X[6] * t[0];  // X[6] is scale velocity
    Y[4] = s * Y[4] + X[6] * t[1];
    Y[5] = s * Y[5] + X[6] * t[2];
    
    Y[6] = X[6];  // Scale velocity remains unchanged under adjoint
}

__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

__device__ void 
relSim3(const float *ti, const float *qi, const float si, const float *tj, const float *qj, const float sj, 
        float *tij, float *qij, float *sij) {
    qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1];
    qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2];
    qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0];
    qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2];

    *sij = sj / si;

    float scaled_ti[3];
    actSO3(qij, ti, scaled_ti);

    tij[0] = tj[0] - (*sij) * scaled_ti[0];
    tij[1] = tj[1] - (*sij) * scaled_ti[1];
    tij[2] = tj[2] - (*sij) * scaled_ti[2];
}

__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}

__global__ void projective_transform2_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float intrinsics_[4];
  __shared__ float fx;
  __shared__ float fy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    intrinsics_[0] = intrinsics[0][0];
    intrinsics_[1] = intrinsics[0][1];
    intrinsics_[2] = intrinsics[0][2];
    intrinsics_[3] = intrinsics[0][3];
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  // jacobians
  float Jp[6];
  float Jz;

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);

    const float x = Xj[0];
    const float y = Xj[1];
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
    const float ru = target[block_id][0][i][j] - xnyn.x;
    const float rv = target[block_id][1][i][j] - xnyn.y;

    // assume pinhole
    Jp[0] = fx * d;
    Jp[1] = 0;
    Jp[2] = fx * (-x * d2);
    Jp[3] = 0;
    Jp[4] = fy * d;
    Jp[5] = fy * (-y * d2);

    // x - coordinate
    Jz = Jp[0] * tij[0] + Jp[1] * tij[1] + Jp[2] * tij[2];
    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    // y - coordinate
    Jz = Jp[3] * tij[0] + Jp[4] * tij[1] + Jp[5] * tij[2];
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;
  }
}

__global__ void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float intrinsics_[4];
  __shared__ float fx;
  __shared__ float fy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
    fx = intrinsics[0];
    fy = intrinsics[1];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  // jacobians
  float Jp[6];
  float Jx[12];
  float Jz;

  float* Ji = &Jx[0];
  float* Jj = &Jx[6];

  // hessians
  float hij[12*(12+1)/2];

  float vi[6], vj[6];

  int l;
  for (l=0; l<12*(12+1)/2; l++) {
    hij[l] = 0;
  }

  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);

    const float x = Xj[0];
    const float y = Xj[1];
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
    const float ru = target[block_id][0][i][j] - xnyn.x;
    const float rv = target[block_id][1][i][j] - xnyn.y;

    // assume pinhole
    Jp[0] = fx * d;
    Jp[1] = 0;
    Jp[2] = fx * (-x * d2);
    Jp[3] = 0;
    Jp[4] = fy * d;
    Jp[5] = fy * (-y * d2);

    // x - coordinate
    Jj[0] = Jp[0] * h;
    Jj[1] = Jp[1] * h;
    Jj[2] = Jp[2] * h;
    Jj[3] = -Jp[1] * z + Jp[2] * y;
    Jj[4] =  Jp[0] * z - Jp[2] * x;
    Jj[5] = -Jp[0] * y + Jp[1] * x;

    Jz = Jp[0] * tij[0] + Jp[1] * tij[1] + Jp[2] * tij[2];
    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    if (ix == jx) wu = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wu * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wu * ru * Ji[n];
      vj[n] += wu * ru * Jj[n];

      Eii[block_id][n][k] = wu * Jz * Ji[n];
      Eij[block_id][n][k] = wu * Jz * Jj[n];
    }

    // y - coordinate
    Jj[0] = Jp[3] * h;
    Jj[1] = Jp[4] * h;
    Jj[2] = Jp[5] * h;
    Jj[3] = -Jp[4] * z + Jp[5] * y;
    Jj[4] =  Jp[3] * z - Jp[5] * x;
    Jj[5] = -Jp[3] * y + Jp[4] * x;

    Jz = Jp[3] * tij[0] + Jp[4] * tij[1] + Jp[5] * tij[2];
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;

    if (ix == jx) wv = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wv * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wv * rv * Ji[n];
      vj[n] += wv * rv * Jj[n];

      Eii[block_id][n][k] += wv * Jz * Ji[n];
      Eij[block_id][n][k] += wv * Jz * Jj[n];
    }


  }

  __syncthreads();

  __shared__ float sdata[THREADS];
  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][block_id][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][block_id][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<12; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        if (n<6 && m<6) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=6 && m<6) {
          Hs[1][block_id][m][n-6] = sdata[0];
          Hs[2][block_id][n-6][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-6][m-6] = sdata[0];
          Hs[3][block_id][m-6][n-6] = sdata[0];
        }
      }

      l++;
    }
  }
}

__global__ void projective_transform_sim3_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float intrinsics_[4];
  __shared__ float fx;
  __shared__ float fy;

  __shared__ float ti[3], tj[3], tij[3];  
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si, sj, sij;

  // load intrinsics from global memory
  if (thread_id == 0) {
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
    fx = intrinsics[0];
    fy = intrinsics[1];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id == 0) {
    si = poses[ix][7];
    sj = poses[jx][7];
  }

  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, &sij);
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  // jacobians
  float Jp[6];
  float Jx[14];
  float Jz;

  float* Ji = &Jx[0];
  float* Jj = &Jx[7];

  // hessians
  float hij[14*(14+1)/2];

  float vi[7], vj[7];

  int l;
  for (l=0; l<14*(14+1)/2; l++) {
    hij[l] = 0;
  }

  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    const int i = k / wd;
    const int j = k % wd;

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

    // transform homogenous point
    actSim3(tij, qij, sij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);

    const float x = Xj[0];
    const float y = Xj[1];
    const float z = (Xj[2] < MIN_DEPTH) ? 0.0 : Xj[2];
    const float h = Xj[3];

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];
    const float ru = target[block_id][0][i][j] - xnyn.x;
    const float rv = target[block_id][1][i][j] - xnyn.y;

    // assume pinhole
    Jp[0] = fx * d;
    Jp[1] = 0;
    Jp[2] = fx * (-x * d2);
    Jp[3] = 0;
    Jp[4] = fy * d;
    Jp[5] = fy * (-y * d2);

    // x - coordinate
    Jj[0] = Jp[0] * h;
    Jj[1] = Jp[1] * h;
    Jj[2] = Jp[2] * h;
    Jj[3] = -Jp[1] * z + Jp[2] * y;
    Jj[4] =  Jp[0] * z - Jp[2] * x;
    Jj[5] = -Jp[0] * y + Jp[1] * x;
    Jj[6] = Jp[0] * x + Jp[1] * y + Jp[2] * z;

    Jz = Jp[0] * tij[0] + Jp[1] * tij[1] + Jp[2] * tij[2];  // need to adapt?
    Cii[block_id][k] = wu * Jz * Jz;
    bz[block_id][k] = wu * ru * Jz;

    if (ix == jx) wu = 0;

    adjSim3(tij, qij, sij, Jj, Ji);
    for (int n=0; n<7; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wu * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += wu * ru * Ji[n];
      vj[n] += wu * ru * Jj[n];

      Eii[block_id][n][k] = wu * Jz * Ji[n];
      Eij[block_id][n][k] = wu * Jz * Jj[n];
    }

    // y - coordinate
    Jj[0] = Jp[3] * h;
    Jj[1] = Jp[4] * h;
    Jj[2] = Jp[5] * h;
    Jj[3] = -Jp[4] * z + Jp[5] * y;
    Jj[4] =  Jp[3] * z - Jp[5] * x;
    Jj[5] = -Jp[3] * y + Jp[4] * x;
    Jj[6] = Jp[3] * x + Jp[4] * y + Jp[5] * z;

    Jz = Jp[3] * tij[0] + Jp[4] * tij[1] + Jp[5] * tij[2];  // need to adapt?
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;

    if (ix == jx) wv = 0;

    adjSim3(tij, qij, sij, Jj, Ji);
    for (int n=0; n<7; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wv * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += wv * rv * Ji[n];
      vj[n] += wv * rv * Jj[n];

      Eii[block_id][n][k] += wv * Jz * Ji[n];
      Eij[block_id][n][k] += wv * Jz * Jj[n];
    }


  }

  __syncthreads();

  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][block_id][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][block_id][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }

      l++;
    }
  }
}

__global__ void frame_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
    const float beta) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float intrinsics_[4];

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();


  //points 
  float Xi[4];
  float Xj[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;
  __shared__ float valid[THREADS]; valid[thread_id] = 0;
  __shared__ float total[THREADS]; total[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();


    relSE3(ti, qi, tj, qj, tij, qij);

    float d, du, dv;
    float2 xnyn;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);


      // if (disps[ix][i][j] < 0.01) {
      //   continue;
      // }
      
      // homogenous coordinates
      iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      xnyn = proj(Xj, intrinsics_);
      du = xnyn.x - u;
      dv = xnyn.y - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += beta;
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += beta * d;
        valid[threadIdx.x] += beta;
      }

      // homogenous coordinates
      iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

      Xj[0] = Xi[0] + Xi[3] * tij[0];
      Xj[1] = Xi[1] + Xi[3] * tij[1];
      Xj[2] = Xi[2] + Xi[3] * tij[2];

      xnyn = proj(Xj, intrinsics_);
      du = xnyn.x - u;
      dv = xnyn.y - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += (1 - beta);
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += (1 - beta) * d;
        valid[threadIdx.x] += (1 - beta);
      }
    }

    if (threadIdx.x == 0) {
      int tmp = ix;
      ix = jx;
      jx = tmp;
    }

    __syncthreads();

  }
  __syncthreads(); blockReduce(accum);
  __syncthreads(); blockReduce(total);
  __syncthreads(); blockReduce(valid);

  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
  }
}

__global__ void covis_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx1, jx2;

  __shared__ float intrinsics_[4];

  __shared__ float ti[3], tj1[3], tij1[3], tj2[3], tij2[3];
  __shared__ float qi[4], qj1[4], qij1[4], qj2[4], qij2[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx1 = ix - 1;
    jx2 = ix + 1;
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj1[4], Xj2[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj1[thread_id] = poses[jx1][thread_id];
      tj2[thread_id] = poses[jx2][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj1[thread_id] = poses[jx1][thread_id+3];
      qj2[thread_id] = poses[jx2][thread_id+3];
    }

    __syncthreads();

    relSE3(ti, qi, tj1, qj1, tij1, qij1);
    relSE3(ti, qi, tj2, qj2, tij2, qij2);

    float2 xnyn1, xnyn2;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);

      // homogenous coordinates
      iproj(u, v, intrinsics_, Xi, disps[ix][i][j]);

      // transform homogenous point
      actSE3(tij1, qij1, Xi, Xj1);
      actSE3(tij2, qij2, Xi, Xj2);

      xnyn1 = proj(Xj1, intrinsics_);
      xnyn2 = proj(Xj2, intrinsics_);
      bool out1 = xnyn1.x < 0 || xnyn1.x >= wd || xnyn1.y < 0 || xnyn1.y >= ht;
      bool out2 = xnyn2.x < 0 || xnyn2.x >= wd || xnyn2.y < 0 || xnyn2.y >= ht;
      if (out1 && out2) {
        accum[threadIdx.x] += 1;
      }
    }

    __syncthreads();
  }
  __syncthreads(); blockReduce(accum);
  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = accum[0] / (ht*wd);
  }
}



__global__ void depth_filter_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter)
{

  const int block_id = blockIdx.x;
  const int neigh_id = blockIdx.y;
  const int index = blockIdx.z * blockDim.x + threadIdx.x;

  // if (threadIdx.x == 0) {
  //   printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x);
  // }

  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float intrinsics_[4];

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  if (threadIdx.x == 0) {
    ix = static_cast<int>(inds[block_id]);
    jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();

  if (jx < 0 || jx >= num) {
    return;
  }

  const float t = thresh[block_id];

  // load poses from global memory
  if (threadIdx.x < 3) {
    ti[threadIdx.x] = poses[ix][threadIdx.x];
    tj[threadIdx.x] = poses[jx][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    qi[threadIdx.x] = poses[ix][threadIdx.x+3];
    qj[threadIdx.x] = poses[jx][threadIdx.x+3];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];
  float2 xnyn;

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[ix][i][j];
    
    // homogenous coordinates
    iproj(ui, vi, intrinsics_, Xi, di);

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    xnyn = proj(Xj, intrinsics_);
    const float uj = xnyn.x;
    const float vj = xnyn.y;
    const float dj = Xj[3] / Xj[2];

    const int u0 = static_cast<int>(floor(uj));
    const int v0 = static_cast<int>(floor(vj));

    if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
      const float wx = ceil(uj) - uj;
      const float wy = ceil(vj) - vj;

      const float d00 = disps[jx][v0+0][u0+0];
      const float d01 = disps[jx][v0+0][u0+1];
      const float d10 = disps[jx][v0+1][u0+0];
      const float d11 = disps[jx][v0+1][u0+1];

      const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

      const float err = abs(1.0/dj - 1.0/dj_hat);
      if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
    }
  }
}



__global__ void iproj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;


  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ float intrinsics_[4];

  __shared__ float t[3];
  __shared__ float q[4];

  if (threadIdx.x == 0) {
    intrinsics_[0] = intrinsics[0];
    intrinsics_[1] = intrinsics[1];
    intrinsics_[2] = intrinsics[2];
    intrinsics_[3] = intrinsics[3];
  }

  __syncthreads();


  // load poses from global memory
  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[block_id][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[block_id][threadIdx.x+3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[block_id][i][j];
    
    // homogenous coordinates
    iproj(ui, vi, intrinsics_, Xi, di);

    // transform homogenous point
    actSE3(t, q, Xi, Xj);

    points[block_id][i][j][0] = Xj[0] / Xj[3];
    points[block_id][i][j][1] = Xj[1] / Xj[3];
    points[block_id][i][j][2] = Xj[2] / Xj[3];

  }
}

__global__ void bi_inter_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> scales,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> grids,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> outputs,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Jacobis)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;

  const int num = grids.size(0);
  const int ht = grids.size(1);
  const int wd = grids.size(2);

  __syncthreads();

  if (index < ht*wd) {
    const int i = index / wd;
    const int j = index % wd;

    const float xind = grids[block_id][i][j][0];
    const float yind = grids[block_id][i][j][1];

    int x0 = floor(xind);
    int y0 = floor(yind);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wa = (x1 - xind) * (y1 - yind);
    float wb = (x1 - xind) * (yind - y0);
    float wc = (xind - x0) * (y1 - yind);
    float wd = (xind - x0) * (yind - y0);

    outputs[block_id][i][j] = wa * scales[block_id][y0][x0] + wb * scales[block_id][y1][x0] + wc * scales[block_id][y0][x1] + wd * scales[block_id][y1][x1];
    Jacobis[block_id][i][j][y0][x0] = wa;
    Jacobis[block_id][i][j][y1][x0] = wb;
    Jacobis[block_id][i][j][y0][x1] = wc;
    Jacobis[block_id][i][j][y1][x1] = wd;
  }
}



__global__ void accum_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  
  const int block_id = blockIdx.x;
  const int D = inps.size(2);

  const int start = ptrs[block_id];
  const int end = ptrs[block_id+1];

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float x = 0;
    for (int i=start; i<end; i++) {
      x += inps[idxs[i]][k];
    }
    outs[block_id][k] = x;
  }  
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}


__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int t0, const int t1) 
{

  for (int k=t0+threadIdx.x; k<t1; k+=blockDim.x) {
    float xi[6], q[4], q1[4], t[3], t1[3];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];
    
    for (int n=0; n<6; n++) {
      xi[n] = dx[k-t0][n];
    }

    retrSE3(xi, t, q, t1, q1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];
  }
}

__global__ void disp_retr_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds) 
{
  const int i = inds[blockIdx.x];
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
    disps[i][k/wd][k%wd] = d;
  }
}

torch::Tensor accum_cuda(torch::Tensor data, torch::Tensor ix, torch::Tensor jx) {
  torch::Tensor ix_cpu = ix.to(torch::kCPU);
  torch::Tensor jx_cpu = jx.to(torch::kCPU);
  torch::Tensor inds = torch::argsort(ix_cpu);

  long* ix_data = ix_cpu.data_ptr<long>();
  long* jx_data = jx_cpu.data_ptr<long>();
  long* kx_data = inds.data_ptr<long>();

  int count = jx.size(0);
  std::vector<int> cols;

  torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
    torch::TensorOptions().dtype(torch::kInt64));
  
  long* ptrs_data = ptrs_cpu.data_ptr<long>();
  ptrs_data[0] = 0;

  int i = 0;
  for (int j=0; j<count; j++) {
    while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
      if (ix_data[kx_data[i]] == jx_data[j])
        cols.push_back(kx_data[i]);
      i++;
    }
    ptrs_data[j+1] = cols.size();
  }

  torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
    torch::TensorOptions().dtype(torch::kInt64));

  long* idxs_data = idxs_cpu.data_ptr<long>();

  for (int i=0; i<cols.size(); i++) {
    idxs_data[i] = cols[i];
  }

  torch::Tensor ptrs = ptrs_cpu.to(torch::kCUDA);
  torch::Tensor idxs = idxs_cpu.to(torch::kCUDA);

  torch::Tensor out = torch::zeros({jx.size(0), data.size(1)},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  accum_kernel<<<count, THREADS>>>(
    data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ptrs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    idxs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    out.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  return out;
}


__global__ void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{

  // indicices
  const int ix = idx[blockIdx.x][0];
  const int jx = idx[blockIdx.x][1];
  const int kx = idx[blockIdx.x][2];

  const int d = E.size(1);
  const int D = E.size(2);

  float dS[7][7];
  float ei[7];
  float ej[7];

  for (int i=0; i<d; i++) {
    for (int j=0; j<d; j++) {
      dS[i][j] = 0;
    }
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q = Q[kx][k];
      
    // coalesced memory read
    for (int n=0; n<d; n++) {
      ei[n] = E[ix][n][k] * q;
      ej[n] = E[jx][n][k];
    }

    // block EEt
    for (int n=0; n<d; n++) {
      for (int m=0; m<d; m++) {
        dS[n][m] += ei[n] * ej[m];
      }
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<d; n++) {
    for (int m=0; m<d; m++) {
      sdata[threadIdx.x] = dS[n][m];

      blockReduce(sdata);

      if (threadIdx.x == 0) {
        S[blockIdx.x][n][m] = sdata[0];
      }
    }
  }
}


__global__ void Ev6x1_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float, 2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
  const int d = E.size(1);
  const int D = E.size(2);
  const int kx = idx[blockIdx.x][0];

  float b[7];
  for (int n=0; n<d; n++) {
    b[n] = 0.0;
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q_w = Q[kx][k] * w[kx][k];

    for (int n=0; n<d; n++) {
      b[n] += q_w * E[blockIdx.x][n][k];
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<d; n++) {
    sdata[threadIdx.x] = b[n];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      v[blockIdx.x][n] += sdata[0];
    }
  }
}

__global__ void EvT6x1_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idx,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w)
{
  const int d = E.size(1);
  const int D = E.size(2);
  const int ix = idx[blockIdx.x];

  if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
    return;

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float dw = 0;
    for (int n=0; n<d; n++) {
      dw += E[blockIdx.x][n][k] * x[ix][n];
    }
    w[blockIdx.x][k] = dw;
  }
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};


SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int t0,
                        const int t1)
{

  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int D = E.size(1);
  const int P = t1 - t0;
  const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= t0 && j <= t1) {
      const int t = j - t0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx, jdx;

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

  torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_cuda.size(0), D, D}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_cuda.size(0), D},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  EEt6x6_kernel<<<ix_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    S.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  Ev6x1_kernel<<<jx_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    v.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // schur block
  SparseBlock A(P, D);
  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - t0);

  return A;
}

std::vector<torch::Tensor> proj_trans_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor ii,
    torch::Tensor jj)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = torch::_unique(ii, true, true);
  torch::Tensor kx = std::get<0>(kuniq);

  // initialize buffers
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  projective_transform2_kernel<<<num, THREADS>>>(
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  torch::Tensor C = accum_cuda(Cii, ii, kx);
  torch::Tensor w = accum_cuda(wi, ii, kx);

  return {C, w};
}


std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep,
    const bool motion_only)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = 
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq);
  torch::Tensor kk_exp = std::get<1>(kuniq);

  torch::Tensor dx;
  torch::Tensor dz;
  torch::Tensor Linv, dzcov;

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);
  torch::Tensor vs = torch::zeros({2, num, 6}, opts);
  torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  for (int itr=0; itr<iterations; itr++) {

    projective_transform_kernel<<<num, THREADS>>>(
      targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());


    // pose x pose block
    SparseBlock A(t1 - t0, 6);

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - t0, 
        torch::cat({ii, jj, ii, jj}) - t0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - t0);

    if (motion_only) {
      dx = A.solve(lm, ep);

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);
    }
    
    else {
      torch::Tensor C = accum_cuda(Cii, ii, kx);
      torch::Tensor w = accum_cuda(wi, ii, kx);
      torch::Tensor Q = 1.0 / (C + eta.view({-1, ht*wd}));

      torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1);
      dx = (A - S).solve(lm, ep);

      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel<<<ix.size(0), THREADS>>>(
        E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

      dz = Q * (w - accum_cuda(dw, ii_exp, kx));

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);

      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
    }

  }

  return {dx, dz, dzcov};
}


std::vector<torch::Tensor> pgba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor Hsp,
    torch::Tensor vsp,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor iip,
    torch::Tensor jjp,
    const int t0,
    const int t1,
    const float lm,
    const float ep)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);
  const int D = 7;

  torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = 
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq);
  torch::Tensor kk_exp = std::get<1>(kuniq);

  torch::Tensor dx, dz;
  torch::Tensor Linv, dzcov;

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, D, D}, opts);
  torch::Tensor vs = torch::zeros({2, num, D}, opts);
  torch::Tensor Eii = torch::zeros({num, D, ht*wd}, opts);
  torch::Tensor Eij = torch::zeros({num, D, ht*wd}, opts);
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);

  projective_transform_sim3_kernel<<<num, THREADS>>>(
    targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    wi.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // pose x pose block
  SparseBlock A(t1 - t0, D);

  // add constraints
  torch::Tensor iii = torch::arange(0, t1-1).to(torch::kCUDA);
  torch::Tensor jji = iii + 1;
  torch::Tensor Hs_all = torch::cat({Hs.reshape({-1,D,D}), Hsp.reshape({-1,D,D})});
  torch::Tensor vs_all = torch::cat({vs.reshape({-1,D}), vsp.reshape({-1,D})});
  torch::Tensor ind1_all = torch::cat({ii, ii, jj, jj, iip, iip, jjp, jjp});
  torch::Tensor ind2_all = torch::cat({ii, jj, ii, jj, iip, jjp, iip, jjp});
  torch::Tensor ind3_all = torch::cat({ii, jj, iip, jjp});

  A.update_lhs(Hs_all, ind1_all - t0, ind2_all - t0);
  A.update_rhs(vs_all, ind3_all - t0);

  // solve system
  torch::Tensor C = accum_cuda(Cii, ii, kx);
  torch::Tensor w = accum_cuda(wi, ii, kx);
  torch::Tensor Q = 1.0 / (C + eta.view({-1, ht*wd}));

  torch::Tensor Ei = accum_cuda(Eii.view({num, D*ht*wd}), ii, ts).view({t1-t0, D, ht*wd});
  torch::Tensor E = torch::cat({Ei, Eij}, 0);

  SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1);
  SparseBlock B = A - S;
  dx = B.solve(lm, ep);
  torch::Tensor ix = jj_exp - t0;
  torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

  EvT6x1_kernel<<<ix.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  dz = Q * (w - accum_cuda(dw, ii_exp, kx));

  // update disparity maps
  disp_retr_kernel<<<kx.size(0), THREADS>>>(
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());

  return {dx, dz};
}


torch::Tensor frame_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  frame_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta);

  return dist;
}

torch::Tensor covis_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  covis_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>());

  return dist;
}


torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}


torch::Tensor iproj_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}

std::vector<torch::Tensor> bi_inter_cuda(
    torch::Tensor scales,
    torch::Tensor grids)
{

  const int nm = grids.size(0);
  const int ht = grids.size(1);
  const int wd = grids.size(2);
  const int hs = scales.size(1);
  const int ws = scales.size(2);

  auto opts = grids.options();
  torch::Tensor outputs = torch::zeros({nm, ht, wd}, opts);
  torch::Tensor Jacobis = torch::zeros({nm, ht, wd, hs, ws}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  bi_inter_kernel<<<blocks, THREADS>>>(
    scales.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    grids.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    outputs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Jacobis.packed_accessor32<float,5,torch::RestrictPtrTraits>());

  return {outputs, Jacobis};

}
