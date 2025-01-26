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

#include "forward.h"
#include "auxiliary.h"
#include "ssr.h"
#include <cooperative_groups.h>
#include <math.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(
	const int idx,
	const int deg,
	const int max_coeffs,
	const glm::vec3* means,
	glm::vec3 campos,
	const float* shs,
	bool* clamped
) {
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least (equations 33)
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	const int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	bool* clamped,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float3* pos_view,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	uint32_t* tiles_touched,
	const dim3 grid,
	const bool prefiltered,
	const bool cubemap)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	const float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	const float p_w = 1.0f / (p_hom.w + 0.0000001f);
	const float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	const float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	const float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	const float det_inv = 1.f / det;
	const float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };  // Inverse of cov2D

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	const float mid = 0.5f * (cov.x + cov.z);
	const float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	const float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	const float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	const float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// Get the covered tile range by the point tile ids stored in `rect_min` and `rect_max`
	getRect(point_image, my_radius, grid, rect_min, rect_max);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// if (cubemap) { // NOTE: To fix the discontinuity at the cubemap edges
	// 	const float3 dir = {p_orig.x - (*cam_pos).x, p_orig.y - (*cam_pos).y, p_orig.z - (*cam_pos).z};
	// 	depths[idx] = sqrtf(square_norm(dir));
	// } else {
	// 	depths[idx] = p_view.z;
	// }
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	pos_view[idx] = {p_view.x, p_view.y, p_view.z};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);	// The number of covered tiles
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
liteRenderCUDA(
	const int W, int H,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ features,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depth,
	const float* __restrict__ bg_color,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ final_T,
	float* __restrict__ out_color,
	float* __restrict__ out_opacity,
	float* __restrict__ out_depth,
	bool argmax_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0.0f };
	float D = 0.0f;
	float O = 0.0f;
	float max_weight = 0.0f;
	float except_depth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			const float weight = alpha * T;
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;
			}

			D += depth[collected_id[j]] * weight;
			O += weight;

			// peak selection
			if (weight > max_weight) {
				except_depth = depth[collected_id[j]];
				max_weight = weight;
			}

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		if (O > 1e-6) {
			out_depth[pix_id] = argmax_depth ? except_depth : D / O; // peak selection or linear interpolation
		} else {
			out_depth[pix_id] = 0.0f;
		}
		out_opacity[pix_id] = O;
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const int W, int H,
	const float fx, float fy,
	const float* means3D,
	const float* cam_pos,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* viewmatrix,
	const float* __restrict__ features,
	const float* __restrict__ normals,
	const float* __restrict__ albedo,
	const float* __restrict__ roughness,
	const float* __restrict__ metallic,
	const float3* __restrict__ pos_view,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ depth,
	const float* __restrict__ bg_color,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ final_T,
	float* __restrict__ out_color,
	float* __restrict__ out_opacity,
	float* __restrict__ out_depth,
	float* __restrict__ out_normal,
	float* __restrict__ out_normal_view,
	float* __restrict__ out_pos,
	float* __restrict__ out_albedo,
	float* __restrict__ out_roughness,
	float* __restrict__ out_metallic,
	bool argmax_depth,
	bool inference)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	float cx = float(W) / 2.0f, cy = float(H) / 2.0f;
	const float3 ray = {(pixf.x - cx) / fx, (pixf.y - cy) / fy, 1.0f};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0.0f };
	float N[CHANNELS] = { 0.0f };
	float A[CHANNELS] = { 0.0f };
	float R = 0.0f;
	float M = 0.0f;
	float D = 0.0f;
	float3 POS = {0.0f, 0.0f, 0.0f};
	float3 N_world = {0.0f, 0.0f, 0.0f};
	float3 N_view = {0.0f, 0.0f, 0.0f};
	float O = 0.0f;
	float max_weight = 0.0f;
	float except_depth = 0.0f;
	float3 except_pos = {0.0f, 0.0f, 0.0f};


	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];  // Get the idx of 3D Gaussian
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) {
				done = true;
				continue;
			}

			float3 view_dir = {
				cam_pos[0] - means3D[collected_id[j] * 3 + 0],
				cam_pos[1] - means3D[collected_id[j] * 3 + 1],
				cam_pos[2] - means3D[collected_id[j] * 3 + 2],
			};
			const float NoV = normals[collected_id[j] * 3 + 0] * view_dir.x + \
							  normals[collected_id[j] * 3 + 1] * view_dir.y + \
							  normals[collected_id[j] * 3 + 2] * view_dir.z;

			const float weight = alpha * T;
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;
				A[ch] += albedo[collected_id[j] * CHANNELS + ch] * weight;
                //if (NoV > 0.0f) // NOTE: the trick from GIR, do not make scene for scenes
				N[ch] += normals[collected_id[j] * CHANNELS + ch] * weight;
			}
			R += roughness[collected_id[j]] * weight;
			M += metallic[collected_id[j]] * weight;



			D += depth[collected_id[j]] * weight;
			POS.x += pos_view[collected_id[j]].x * weight;
			POS.y += pos_view[collected_id[j]].y * weight;
			POS.z += pos_view[collected_id[j]].z * weight;
			O += weight;

			if (weight > max_weight) {
				except_depth = depth[collected_id[j]];
				except_pos.x = pos_view[collected_id[j]].x;
				except_pos.y = pos_view[collected_id[j]].y;
				except_pos.z = pos_view[collected_id[j]].z;
				max_weight = weight;
			}

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;

		N_world = {N[0], N[1], N[2]};
		N_view = transformVec4x3(N_world, viewmatrix);
		N_view = normalize(N_view);
		out_normal_view[pix_id] = N_view.x;
		out_normal_view[1 * H * W + pix_id] = N_view.y;
		out_normal_view[2 * H * W + pix_id] = N_view.z;
		
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
			out_normal[ch * H * W + pix_id] = N[ch];
			out_albedo[ch * H * W + pix_id] = A[ch];
		}
		if (inference) {
			out_roughness[pix_id] = R + T;
		} else {
			out_roughness[pix_id] = R;
		}
		out_metallic[pix_id] = M;
		if (O > 1e-6) {
			out_depth[pix_id] = argmax_depth ? except_depth : D / O;
			out_pos[pix_id] = argmax_depth ? except_pos.x : POS.x / O;
			out_pos[1 * H * W + pix_id] = argmax_depth ? except_pos.y : POS.y / O;
			out_pos[2 * H * W + pix_id] = argmax_depth ? except_pos.z : POS.z / O;
		} else {
			out_depth[pix_id] = 0.0f;
			out_pos[pix_id] = 0.0f;
			out_pos[1 * H * W + pix_id] = 0.0f;
			out_pos[2 * H * W + pix_id] = 0.0f;
		}
		

		out_opacity[pix_id] = O;
	}
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
SSAOCUDA(
	int W, int H,
	const float focal_x,
	const float focal_y,
	const float radius,  //0.8
	const float bias, //-0.01
	const float thick, //-0.05
	const float delta, //0.0625
	const int step, //16
	const int start, //8
	const float* __restrict__ out_normal,
	const float* __restrict__ out_pos,
	float* __restrict__ occlusion)
{
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	if (pix.x > W-1 || pix.y > H-1)
		return;

	float3 normal_un = {out_normal[pix_id], out_normal[1 * H * W + pix_id], out_normal[2 * H * W + pix_id]};
	float3 normal = normalize(normal_un);
	float3 pos = {out_pos[pix_id], out_pos[1 * H * W + pix_id], out_pos[2 * H * W + pix_id]};
	float3 up = {0.0f, 1.0f, 0.0f};
	float rndot = dot(up, normal); 
	float3 untangent = {up.x - normal.x * rndot, up.y - normal.y * rndot, up.z - normal.z * rndot};
	float3 tangent = normalize(untangent);
	float3 bitangent = normalize(cross(normal, tangent));
	float TBN[9];
	TBN[0] = tangent.x;
	TBN[1] = tangent.y;
	TBN[2] = tangent.z;
	TBN[3] = bitangent.x;
	TBN[4] = bitangent.y;
	TBN[5] = bitangent.z;
	TBN[6] = normal.x;
	TBN[7] = normal.y;
	TBN[8] = normal.z;
	float occ = 0.0;
	float sampleDelta = delta * M_PIf;
    float nrSamples = 0.0; 
    for(float phi = 0.0; phi < 2.0 * M_PIf; phi += sampleDelta)
    {
        for(float theta = 0.0; theta <= 0.5 * M_PIf; theta += sampleDelta * 0.5)
        {
        // spherical to cartesian (in tangent space)
			float cosh = cosf(theta);
            float3 tangentSample = {sinf(theta) * cosf(phi),  sinf(theta) * sinf(phi), cosf(theta)};
            tangentSample = normalize(tangentSample);
        // tangent space to view
            float3 sampleVec = transformVec3x3(tangentSample, TBN);
            float3 samplePos = {0.0f, 0.0f, 0.0f};
			nrSamples += cosh * sinf(theta);
		    for(int j = start; j < step; ++j)
		    {
			    samplePos.x = pos.x + sampleVec.x * j * (1 + pos.z / 100) * (1 + pos.z / 100 ) * radius / step; //100=zfar-znear
			    samplePos.y = pos.y + sampleVec.y * j * (1 + pos.z / 100) * (1 + pos.z / 100)* radius / step; 
			    samplePos.z = pos.z + sampleVec.z * j * (1 + pos.z / 100) * (1 + pos.z / 100) * radius / step; 
			    float cx = float(W) / 2.0f, cy = float(H) / 2.0f;
			    int2 depth_id = get_coord(cx, cy, focal_x, focal_y, samplePos);
			    if (depth_id.x < 0)
				    break;
			    else if (depth_id.x > W - 1)
				    break;
			    if (depth_id.y < 0)
				    break;
			    else if (depth_id.y > H - 1)
				    break;
				float sampleDepth = out_pos[2 * H * W + W * depth_id.y + depth_id.x]; 
				

			    if (sampleDepth <= samplePos.z + bias && sampleDepth >= samplePos.z - thick) 
			    {
				    occ += cosh * sinf(theta);
				    break;
			    }
		    }
        }
    }
	if(nrSamples > 0.0){
		occlusion[pix_id] = fmaxf(0.0f, fminf(1.0f, 1.0 - (occ / nrSamples)));

	}
    else{
		occlusion[pix_id] = 1.0;
	}
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
SSRCUDA(
	int W, int H,
	const float focal_x,
	const float focal_y,
	const float radius,  //0.8
	const float bias, //-0.01
	const float thick, //-0.05
	const float delta, //0.0625
	const int step, //16
	const int start, //8
	const float* __restrict__ out_normal,
	const float* __restrict__ out_pos,
	const float* __restrict__ out_rgb,
    const float* __restrict__ out_albedo,
    const float* __restrict__ out_roughness,
    const float* __restrict__ out_metallic,
    const float* __restrict__ out_F0,
	float* __restrict__ color,
	float* __restrict__ abd)
{
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	if (pix.x > W-1 || pix.y > H-1)
		return;
    float3 pos = {out_pos[pix_id], out_pos[1 * H * W + pix_id], out_pos[2 * H * W + pix_id]};

    float3 diffuse = {0.0f, 0.0f, 0.0f}; 
	float3 gd = {0.0f, 0.0f, 0.0f}; 
    float3 specular = {0.0f, 0.0f, 0.0f};                                                                                                 
	float3 normal_un = {out_normal[pix_id], out_normal[1 * H * W + pix_id], out_normal[2 * H * W + pix_id]};
	float3 normal = normalize(normal_un);
    float3 N = normal;
    float3 up = {0.0f, 1.0f, 0.0f};
    float rndot = dot(up, normal); 
	float3 untangent = {up.x - normal.x * rndot, up.y - normal.y * rndot, up.z - normal.z * rndot};
	float3 tangent = normalize(untangent);
	float3 bitangent = normalize(cross(normal, tangent));
    float TBN[9];
	TBN[0] = tangent.x;
	TBN[1] = tangent.y;
	TBN[2] = tangent.z;
	TBN[3] = bitangent.x;
	TBN[4] = bitangent.y;
	TBN[5] = bitangent.z;
	TBN[6] = normal.x;
	TBN[7] = normal.y;
	TBN[8] = normal.z;


    float3 albedo = {out_albedo[pix_id], out_albedo[1 * H * W + pix_id], out_albedo[2 * H * W + pix_id]};
    float3 F0 = {out_F0[pix_id], out_F0[1 * H * W + pix_id], out_F0[2 * H * W + pix_id]};
    float roughness = out_roughness[pix_id];
    float metallic = out_metallic[pix_id];

    float3 V = normalize(-pos);
    float3 F = fresnelSchlick(fmaxf(dot(N, V), 0.0000001), F0);
    float3 kS = F;
    float3 kD = {1.0 - kS.x, 1.0 - kS.y, 1.0 - kS.z};
    kD.x *= 1.0 - metallic;
    kD.y *= 1.0 - metallic;
    kD.z *= 1.0 - metallic;

    float sampleDelta = delta * M_PIf;
    float nrSamples = 0.0; 
    for(float phi = 0.0; phi < 2.0 * M_PIf; phi += sampleDelta)
    {
        for(float theta = 0.0; theta <= 0.5 * M_PIf; theta += sampleDelta * 0.5)
        {
        // spherical to cartesian (in tangent space)
            float3 tangentSample = {sinf(theta) * cosf(phi),  sinf(theta) * sinf(phi), cosf(theta)};
            tangentSample = normalize(tangentSample);
        // tangent space to view
            float3 sampleVec = transformVec3x3(tangentSample, TBN);
            float3 samplePos = {0.0f, 0.0f, 0.0f};
			nrSamples += 1;
		    for(int j = start; j < step; ++j)
		    {
			    samplePos.x = pos.x + sampleVec.x * j * (1 + pos.z / 100) * (1 + pos.z / 100) * radius / step; 
			    samplePos.y = pos.y + sampleVec.y * j * (1 + pos.z / 100) * (1 + pos.z / 100)* radius / step; 
			    samplePos.z = pos.z + sampleVec.z * j * (1 + pos.z / 100) * (1 + pos.z / 100) * radius / step; 
			    float cx = float(W) / 2.0f, cy = float(H) / 2.0f;
			    int2 depth_id = get_coord(cx, cy, focal_x, focal_y, samplePos);
			    if (depth_id.x < 0)
				    break;
			    else if (depth_id.x > W - 1)
				    break;
			    if (depth_id.y < 0)
				    break;
			    else if (depth_id.y > H - 1)
				    break;
			    float3 rgb = {out_rgb[W * depth_id.y + depth_id.x], out_rgb[H * W + W * depth_id.y + depth_id.x], out_rgb[2 * H * W + W * depth_id.y + depth_id.x]}; 
				float sampleDepth = out_pos[2 * H * W + W * depth_id.y + depth_id.x]; 
			    if (sampleDepth <= samplePos.z + bias && sampleDepth >= samplePos.z - thick)  //0.05 0.1
			    {
				    diffuse.x += rgb.x * cosf(theta) * sinf(theta);
                    diffuse.y += rgb.y * cosf(theta) * sinf(theta);
                    diffuse.z += rgb.z * cosf(theta) * sinf(theta);
                    // nrSamples++;
				    break;
			    }
		    }
        }
    }
	if(nrSamples > 0.0){
		gd.x = M_PIf * diffuse.x * (1.0 / float(nrSamples)) * kD.x;  //calculate gradient in forward pass, no backward.
		gd.y = M_PIf * diffuse.y * (1.0 / float(nrSamples)) * kD.y;
    	gd.z = M_PIf * diffuse.z * (1.0 / float(nrSamples)) * kD.z;
		diffuse.x = gd.x * albedo.x;
		diffuse.y = gd.y * albedo.y;
    	diffuse.z = gd.z * albedo.z;
	}
    else{
		diffuse.x = 0.0000001;
		diffuse.y = 0.0000001;
    	diffuse.z = 0.0000001;
		gd.x = 0.0000001;
		gd.y = 0.0000001;
    	gd.z = 0.0000001;
	}
   

//------------------------- indlight for specular component, you can modify it if you are interested------//
	// nrSamples = 0.0; 
    // const uint SAMPLE_COUNT = 64;      
    // for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    // {
    //     float3 samplePos = {0.0f, 0.0f, 0.0f};
    //     float2 Xi = Hammersley(i, SAMPLE_COUNT);
    //     float3 Half = ImportanceSampleGGX(Xi, N, roughness);
    //     float3 L = normalize(2.0 * dot(V, Half) * Half - V);
    //     float NdotL = fmaxf(dot(N, L), 0.0);
    //     if(NdotL > 0.0)
    //     {
    //         for(int j = 4; j < step; ++j)
	// 	    {
	// 		    samplePos.x = pos.x + L.x * j * (1 + pos.z / 100) * (1 + pos.z / 100 ) * radius / step; 
	// 		    samplePos.y = pos.y + L.y * j * (1 + pos.z / 100) * (1 + pos.z / 100)* radius / step; 
	// 		    samplePos.z = pos.z + L.z * j * (1 + pos.z / 100) * (1 + pos.z / 100) * radius / step; 
	// 		    float cx = float(W) / 2.0f, cy = float(H) / 2.0f;
	// 		    int2 depth_id = get_coord(cx, cy, focal_x, focal_y, samplePos);
	// 		    if (depth_id.x < 0)
	// 			    break;
	// 		    else if (depth_id.x > W - 1)
	// 			    break;
	// 		    if (depth_id.y < 0)
	// 			    break;
	// 		    else if (depth_id.y > H - 1)
	// 			    break;
	// 		    float3 rgb = {out_rgb[W * depth_id.y + depth_id.x], out_rgb[H * W + W * depth_id.y + depth_id.x], out_rgb[2 * H * W + W * depth_id.y + depth_id.x]}; 
	// 			float sampleDepth = out_pos[2 * H * W + W * depth_id.y + depth_id.x]; 
	// 		    if (sampleDepth <= samplePos.z + bias && sampleDepth >= samplePos.z - 0.15)
	// 		    {
				    
    //                 float attenuation = 1.0 / ((samplePos.x-pos.x)*(samplePos.x-pos.x)+(samplePos.y-pos.y)*(samplePos.y-pos.y)+(samplePos.z-pos.z)*(samplePos.z-pos.z)+0.0001);
    //                 float3 radiance = {rgb.x * attenuation, rgb.y * attenuation, rgb.z * attenuation};
    //                 float NDF = DistributionGGX(N, Half, roughness);        
    //                 float G = GeometrySmith(N, V, L, roughness);      
    //                 float3 nominator = {NDF * G * F.x, NDF * G * F.y, NDF * G * F.z};
    //                 float denominator = 4.0 * fmaxf(dot(N, V), 0.0) * fmaxf(dot(N, L), 0.0) + 0.001; 
    //                 float3 spec = {nominator.x / denominator, nominator.y / denominator, nominator.z / denominator};               
    //                 specular.x += spec.x * radiance.x * NdotL; 
    //                 specular.y += spec.y * radiance.y * NdotL; 
    //                 specular.z += spec.z * radiance.z * NdotL; 
	// 				nrSamples++;
	// 			    break;
	// 		    }
	// 	    }
    //     }
    // }
    color[pix_id] = diffuse.x;
    color[1 * H * W + pix_id] = diffuse.y;
    color[2 * H * W + pix_id] = diffuse.z;	

	abd[pix_id] = gd.x;
    abd[1 * H * W + pix_id] = gd.y;
    abd[2 * H * W + pix_id] = gd.z;
	// color[pix_id] = diffuse.x + specular.x * (1.0 / float(nrSamples));
    // color[1 * H * W + pix_id] = diffuse.y + specular.y * (1.0 / float(nrSamples));
    // color[2 * H * W + pix_id] = diffuse.z + specular.z * (1.0 / float(nrSamples));	
}




__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
depthmapToNormalCUDA(
	int W, int H,
	const float focal_x,
	const float focal_y,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ out_depth,
	float* __restrict__ normal_from_depth,
	float* __restrict__ depth_pos)
{
	// Identify current tile and associated min/max pixel range.

	
	auto block = cg::this_thread_block();
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;

	// if (pix.x > W-1 || pix.y > H-1)
	// 	return;
	// Check if this thread is associated with a valid pixel or outside.
	if (pix.x <= 0 || pix.x >= W - 1 || pix.y <= 0 || pix.y >= H - 1) return;


	const float depth_thresh = 0.01f;
	const float depth = out_depth[pix_id];
	float cx = float(W) / 2.0f, cy = float(H) / 2.0f;
	float3 pos = get_position(pix.x, pix.y, cx, cy, focal_x, focal_y, depth);
	depth_pos[pix_id] = pos.x;
	depth_pos[1 * H * W + pix_id] = pos.y;
	depth_pos[2 * H * W + pix_id] = pos.z;

	if (depth < depth_thresh) return;

	// int pad = 2;
	// for (int x = -pad; x < pad + 1; ++x) {
	// 	if (int(pix.x + x) < 0 || int(pix.x + x) > W - 1) return;
	// 	for (int y = -pad; y < pad + 1; ++y) {
	// 		if (int(pix.y + y) < 0 || int(pix.y + y) > H - 1) return;
	// 		if (out_depth[pix_id + y + W * x] < depth_thresh) return;
	// 	}
	// }
	// float depth_left = out_depth[pix_id - 1], depth_right = out_depth[pix_id + 1];
	// float depth_up = out_depth[pix_id - W], depth_down = out_depth[pix_id + W];

	// float3 pos_cen = get_position(pix.x, pix.y, cx, cy, focal_x, focal_y, depth);
	// float3 pos_left = get_position(pix.x - 1, pix.y, cx, cy, focal_x, focal_y, depth_left);
	// float3 pos_right = get_position(pix.x + 1, pix.y, cx, cy, focal_x, focal_y, depth_right);
	// float3 pos_up = get_position(pix.x, pix.y - 1, cx, cy, focal_x, focal_y, depth_up);
	// float3 pos_down = get_position(pix.x, pix.y + 1, cx, cy, focal_x, focal_y, depth_down);
	// float3 ddx = fabsf(depth_left - depth) < fabs(depth_right - depth) ? (pos_cen - pos_left) : (pos_right - pos_cen);
	// float3 ddy = fabsf(depth_down - depth) < fabs(depth_up - depth) ? (pos_cen - pos_down) : (pos_up - pos_cen);
	// float3 normal = cross(ddx, ddy);
	// normal = normalize(normal);

	// // NOTE: rotation (it should be c2w!!!)
	// const float normal_x = viewmatrix[0] * normal.x + viewmatrix[1] * normal.y + viewmatrix[2] * normal.z;
	// const float normal_y = viewmatrix[4] * normal.x + viewmatrix[5] * normal.y + viewmatrix[6] * normal.z;
	// const float normal_z = viewmatrix[8] * normal.x + viewmatrix[9] * normal.y + viewmatrix[10] * normal.z;

	// normal_from_depth[0 * H * W + pix_id] = normal_x;
	// normal_from_depth[1 * H * W + pix_id] = normal_y;
	// normal_from_depth[2 * H * W + pix_id] = normal_z;
	// filter out the edge to avoid the noise normal
	int pad = 2;
	for (int x = -pad; x < pad + 1; ++x) {
		if (int(pix.x + x) < 0 || int(pix.x + x) > W - 1) return;
		for (int y = -pad; y < pad + 1; ++y) {
			if (int(pix.y + y) < 0 || int(pix.y + y) > H - 1) return;
			if (out_depth[pix_id + W * y + x] < depth_thresh) return;
		}
	}
	float depth_aa = out_depth[pix_id - W];
	float depth_bb = out_depth[pix_id + 1];
	float depth_cc = out_depth[pix_id + W];
	float depth_dd = out_depth[pix_id - 1];
	float depth_ab = out_depth[pix_id - W + 1];
	float depth_bc = out_depth[pix_id + W + 1];
	float depth_cd = out_depth[pix_id + W - 1];
	float depth_da = out_depth[pix_id - W - 1];

	float3 pos_aa = get_position(pix.x, pix.y - 1, cx, cy, focal_x, focal_y, depth_aa);
	float3 pos_bb = get_position(pix.x + 1, pix.y, cx, cy, focal_x, focal_y, depth_bb);
	float3 pos_cc = get_position(pix.x, pix.y + 1, cx, cy, focal_x, focal_y, depth_cc);
	float3 pos_dd = get_position(pix.x - 1, pix.y, cx, cy, focal_x, focal_y, depth_dd);
	float3 pos_ab = get_position(pix.x + 1, pix.y - 1, cx, cy, focal_x, focal_y, depth_ab);
	float3 pos_bc = get_position(pix.x + 1, pix.y + 1, cx, cy, focal_x, focal_y, depth_bc);
	float3 pos_cd = get_position(pix.x - 1, pix.y + 1, cx, cy, focal_x, focal_y, depth_cd);
	float3 pos_da = get_position(pix.x - 1, pix.y - 1, cx, cy, focal_x, focal_y, depth_da);
	float3 edge_a = pos_da - pos_ab;
	float3 edge_b = pos_ab - pos_bc;
	float3 edge_c = pos_bc - pos_cd;
	float3 edge_d = pos_cd - pos_da;
	float3 edge_ac = pos_cc - pos_aa;
	float3 edge_bd = pos_dd - pos_bb;
	float3 edge_cdab = pos_ab - pos_cd;
	float3 edge_bcad = pos_da - pos_bc;

	float3 normal1 = cross(edge_a, edge_d);
	float3 normal2 = cross(edge_d, edge_c);
	float3 normal3 = cross(edge_c, edge_b);
	float3 normal4 = cross(edge_b, edge_a);
	float3 normal5 = cross(edge_ac, edge_bd);
	float3 normal6 = cross(edge_bcad, edge_cdab);
	float3 normal = (normalize(normal1) + normalize(normal2) + normalize(normal3) + normalize(normal4) + normalize(normal5) + normalize(normal6))/6;

	// NOTE: rotation (it should be c2w!!!)
	const float normal_x = viewmatrix[0] * normal.x + viewmatrix[1] * normal.y + viewmatrix[2] * normal.z;
	const float normal_y = viewmatrix[4] * normal.x + viewmatrix[5] * normal.y + viewmatrix[6] * normal.z;
	const float normal_z = viewmatrix[8] * normal.x + viewmatrix[9] * normal.y + viewmatrix[10] * normal.z;
	// const float normal_x = normal.x;
	// const float normal_y = normal.y;
	// const float normal_z = normal.z;

	normal_from_depth[pix_id] = normal_x;
	normal_from_depth[1 * H * W + pix_id] = normal_y;
	normal_from_depth[2 * H * W + pix_id] = normal_z;
}

void FORWARD::lite_render(
	const dim3 grid, dim3 block,
	int W, int H,
	const uint2* ranges,
	const uint32_t* point_list,
	const float* colors,
	const float2* means2D,
	const float4* conic_opacity,
	const float* depth,
	const float* bg_color,
	uint32_t* n_contrib,
	float* final_T,
	float* out_color,
	float* out_opacity,
	float* out_depth,
	bool argmax_depth)
{
	liteRenderCUDA<NUM_CHANNELS><<<grid, block>>>(
		W, H,
		ranges,
		point_list,
		colors,
		means2D,
		conic_opacity,
		depth,
		bg_color,
		n_contrib,
		final_T,
		out_color,
		out_opacity,
		out_depth,
		argmax_depth);
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const int W, int H,
	const float fx, float fy,
	const float* means3D,
	const float* cam_pos,
	const uint2* ranges,
	const uint32_t* point_list,
	const float* viewmatrix,
	const float* colors,
	const float* normal,
	const float* albedo,
	const float* roughness,
	const float* metallic,
	const float3* pos_view,
	const float2* means2D,
	const float4* conic_opacity,
	const float* depth,
	const float* bg_color,
	uint32_t* n_contrib,
	float* final_T,
	float* out_color,
	float* out_opacity,
	float* out_depth,
	float* out_normal,
	float* out_normal_view,
	float* out_pos,
	float* out_albedo,
	float* out_roughness,
	float* out_metallic,
	const bool argmax_depth,
	const bool inference)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		W, H,
		fx, fy,
		means3D,
		cam_pos,
		ranges,
		point_list,
		viewmatrix,
		colors,
		normal,
		albedo,
		roughness,
		metallic,
		pos_view,
		means2D,
		conic_opacity,
		depth,
		bg_color,
		n_contrib,
		final_T,
		out_color,
		out_opacity,
		out_depth,
		out_normal,
		out_normal_view, 
		out_pos,
		out_albedo,
		out_roughness,
		out_metallic,
		argmax_depth,
		inference);
}

void FORWARD::preprocess(
	const int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	bool* clamped,
	float2* means2D,
	float* depths,
	float3* pos_view,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	uint32_t* tiles_touched,
	const dim3 grid,
	const bool prefiltered,
	const bool cubemap)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>> (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		clamped,
		radii,
		means2D,
		depths,
		pos_view,
		cov3Ds,
		rgb,
		conic_opacity,
		tiles_touched,
		grid,
		prefiltered,
		cubemap
	);
}

void FORWARD::depthToNormal(
	const dim3 grid,
	const dim3 block,
	const int W, int H,
	const float focal_x,
	const float focal_y,
	const float* viewmatrix,
	const float* depthMap,
	float* normalMap,
	float* normal_from_depth_view) {
	depthmapToNormalCUDA<<<grid, block>>>(
		W, H,
		focal_x,
		focal_y,
		viewmatrix,
		depthMap,
		normalMap,
		normal_from_depth_view
	);
}

void FORWARD::SSAO(
	const dim3 grid, 
	const dim3 block,
	int W, int H,
	const float focal_x,
	const float focal_y,
	const float radius,  //0.8
	const float bias, //-0.01
	const float thick, //-0.05
	const float delta, //0.0625
	const int step, //16
	const int start,
	const float* out_normal,
	const float* out_pos,
	float* occlusion) {
	SSAOCUDA<<<grid, block>>>(
		W, H,
		focal_x,
		focal_y,
		radius,  //0.8
		bias, //-0.01
		thick, //-0.05
		delta, //0.0625
		step, //16
		start,
		out_normal,
		out_pos,
		occlusion
	);
}

void FORWARD::SSR(
	const dim3 grid, 
	const dim3 block,
	int W, int H,
	const float focal_x,
	const float focal_y,
	const float radius,  //0.8
	const float bias, //-0.01
	const float thick, //-0.05
	const float delta, //0.0625
	const int step, //16
	const int start,
	const float* out_normal,
	const float* out_pos,
	const float* out_rgb,
	const float* out_albedo,
    const float* out_roughness,
    const float* out_metallic,
    const float* out_F0,
	float* color,
	float* abd) {
	SSRCUDA<<<grid, block>>>(
		W, H,
		focal_x,
		focal_y,
		radius,  //0.8
		bias, //-0.01
		thick, //-0.05
		delta, //0.0625
		step, //16
		start,
		out_normal,
		out_pos,
		out_rgb,
		out_albedo,
		out_roughness,
		out_metallic,
		out_F0,
		color,
		abd
	);
}

