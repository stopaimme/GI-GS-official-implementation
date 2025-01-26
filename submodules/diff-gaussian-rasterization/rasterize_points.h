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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LiteRasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float scale_modifier,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const bool prefiltered,
	const bool argmax_depth);
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& normal,
    const torch::Tensor& albedo,
    const torch::Tensor& roughness,
    const torch::Tensor& metallic,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float scale_modifier,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const bool prefiltered,
	const bool argmax_depth,
	const bool inference,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
	torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& normal,
    const torch::Tensor& albedo,
    const torch::Tensor& roughness,
    const torch::Tensor& metallic,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float scale_modifier,
	const float tan_fovx, 
	const float tan_fovy,
	const int degree,
	const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_opacity,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& dL_dout_albedo,
    const torch::Tensor& dL_dout_roughness,
    const torch::Tensor& dL_dout_metallic,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int R,
	const bool debug);

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix);

std::tuple<torch::Tensor, torch::Tensor> depthToNormal(
	const int width, const int height,
	const float focal_x,
	const float focal_y,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& depthMap);

torch::Tensor SSAO(
	const int width, const int height,
	const float focal_x,
	const float focal_y,
	const float radius,  //0.8
	const float bias, //-0.01
	const float thick, //-0.05
	const float delta, //0.0625
	const int step, //16
	const int start,
	const torch::Tensor& out_normal,
	const torch::Tensor& out_pos);

std::tuple<torch::Tensor, torch::Tensor> SSR(
	const int width, const int height,
	const float focal_x,
	const float focal_y,
	const float radius,  //0.8
	const float bias, //-0.01
	const float thick, //-0.05
	const float delta, //0.0625
	const int step, //16
	const int start,
	const torch::Tensor& out_normal,
	const torch::Tensor& out_pos,
	const torch::Tensor& out_rgb,
	const torch::Tensor& out_albedo,
	const torch::Tensor& out_roughness,
	const torch::Tensor& out_metallic,
	const torch::Tensor& out_F0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SSR_BACKWARD(
	const int width, const int height,
	const float focal_x,
	const float focal_y,
	const torch::Tensor& out_normal,
	const torch::Tensor& out_pos,
	const torch::Tensor& out_rgb,
	const torch::Tensor& out_albedo,
	const torch::Tensor& out_roughness,
	const torch::Tensor& out_metallic,
	const torch::Tensor& out_F0,
	const torch::Tensor& dL_dpixels);