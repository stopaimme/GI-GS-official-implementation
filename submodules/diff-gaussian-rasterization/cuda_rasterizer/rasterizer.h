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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static void depthToNormal(
			const int width, const int height,
			const float focal_x,
			const float focal_y,
			const float* viewmatrix,
			const float* depthMap,
			float* normalMap,
			float* depth_pos);

		static void SSAO(
			const int width, const int height,
			const float focal_x,
			const float focal_y,
			const float radius,  //0.8
			const float bias, //-0.01
			const float thick, //-0.05
			const float delta, //0.0625
			const int step, //16
			const int start, //8
			const float* out_normal,
			const float* out_pos,
			float* occlusion);

		static void SSR(
			const int width, const int height,
			const float focal_x,
			const float focal_y,
			const float radius,  //0.8
			const float bias, //-0.01
			const float thick, //-0.05
			const float delta, //0.0625
			const int step, //16
			const int start, //8
			const float* out_normal,
			const float* out_pos,
			const float* out_rgb,
			const float* out_albedo,
    		const float* out_roughness,
    		const float* out_metallic,
    		const float* out_F0,
			float* color,
			float* abd);

		static void SSR_BACKWARD(
			const int width, const int height,
			const float focal_x,
			const float focal_y,
			const float* out_normal,
			const float* out_pos,
			const float* out_rgb,
			const float* out_albedo,
    		const float* out_roughness,
    		const float* out_metallic,
    		const float* out_F0,
			const float* dL_dpixels,
			float* dl_albedo,
			float* dl_roughness,
			float* dl_metallic);

		static int lite_forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			const bool argmax_depth,
			float* out_color,
			float* out_opacity,
			float* out_depth,
			int* radii = nullptr);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* normal,
			const float* albedo,
			const float* roughness,
			const float* metallic,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			const bool argmax_depth,
			const bool inference,
			float* out_color,
			float* out_opacity,
			float* out_depth,
			float* out_normal,
			float* out_normal_view,
			float* out_pos,
			float* out_albedo,
			float* out_roughness,
			float* out_metallic,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* normal,
			const float* albedo,
			const float* roughness,
			const float* metallic,
			const float* scales,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const int* radii,
			const float scale_modifier,
			const float tan_fovx, float tan_fovy,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix_depth,
			const float* dL_dpix,
			const float* dL_dpix_opacity,
			const float* dL_dpix_normal,
			const float* dL_dpix_albedo,
			const float* dL_dpix_roughness,
			const float* dL_dpix_metallic,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_depth,
			float* dL_dopacity,
			float* dL_dnormal,
			float* dL_dalbedo,
			float* dL_droughness,
			float* dL_dmetallic,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif