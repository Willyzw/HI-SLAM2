/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,
		float3* view_points,
		float* depths,
		float2* ray_planes,
		float* ts,
		float* cov3Ds,
		float* rgb,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* view_points,
		const float2* means2D,
		const float* colors,
		const float* semantics,
		const float* ts,
		const float2* ray_planes,
		const float4* conic_opacity,
		const float focal_x, float focal_y,
		float* out_alpha,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_semantic,
		float* out_depth,
		float* accum_depth,
		int* n_touched);
}


#endif