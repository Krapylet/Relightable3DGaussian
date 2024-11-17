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

#include <math.h>
#include "utils/includeTorch.cu"
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include <fstream>
#include <string>
#include <functional>
#include "utils/texture.h"
#include "cuda_rasterizer/auxiliary.h"
#include "cuda_rasterizer/postProcessShader.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const float time,
	const float dt,
	const torch::Tensor& means3D, 		//
	const torch::Tensor& features,
    const torch::Tensor& colors,
    const torch::Tensor& opacity, 		//
	const torch::Tensor& scales, 		//
	const torch::Tensor& rotations, 	//
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& shShaderAddresses,
	const torch::Tensor& splatShaderAddresses,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& viewmatrix_inv,
	const torch::Tensor& projmatrix,
	const torch::Tensor& projmatrix_inv,
	const float tan_fovx, 
	const float tan_fovy,
	const float cx,
	const float cy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh, 			
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool computer_pseudo_normal,
	const int64_t d_textureManager_ptr, // is actually a TextureManager* stored on device.
	const std::vector<int64_t> postProcessingPasses_ptr, // is actually a vector of PostProcessShaders
	const bool debug)
{
	// cast the texture manager back into its original class.
	auto d_textureManager = (Texture::TextureManager *const)d_textureManager_ptr;

	// We can't cast the vector to the correct type directly, so we do it in a hacky way instead
	auto ppArray = (PostProcess::PostProcessShader*) &postProcessingPasses_ptr[0];
	auto postProcessingPasses = std::vector<PostProcess::PostProcessShader>(ppArray, ppArray + postProcessingPasses_ptr.size());

	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	
	const int P = means3D.size(0);
	const int S = features.size(1);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({H, W, NUM_CHANNELS}, 0.0, float_opts);
	torch::Tensor out_opacity = torch::full({H, W, 1}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({H, W, 1}, 0.0, float_opts);
	torch::Tensor out_stencil = torch::full({H, W, 1}, 0.0, float_opts);
	torch::Tensor out_feature = torch::full({H, W, S}, 0.0, float_opts);
	torch::Tensor out_shader_color = torch::full({H, W, NUM_CHANNELS}, 0.0, float_opts);
	torch::Tensor out_normal = torch::full({H, W, 3}, 0.0, float_opts);
	torch::Tensor out_surface_xyz = torch::full({H, W, 3}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	//std::cout << "At ShDefault Cracks texture: " << rawTextures.at("ShDefault").at("Cracks") ///  should be .at("rawData")  /// .cpu().contiguous().data_ptr<float>()[0] << ", At ShDefault Red texture:" << rawTextures.at("ShDefault").at("Red").cpu().contiguous().data_ptr<float>()[0] << std::endl;

	// Since the addresses used for these arrays point to the same memory as used in the python frontend, any changes we make to them will stay permanent.
	// While this is an interesting feature (that should maybe be toggleable?) we don't want that right now. We therefore have to copy
	// every array that contains a value we want to be able to change non-persistantly
	torch::Tensor temp_means3D = means3D.detach().clone();
	torch::Tensor temp_features = features.detach().clone();
	torch::Tensor temp_opacity = opacity.detach().clone();
	torch::Tensor temp_scales = scales.detach().clone();
	torch::Tensor temp_rotations = rotations.detach().clone();
	torch::Tensor temp_sh = sh.detach().clone();

	int rendered = 0;
	if(P != 0)
	{
		int M = 0;
		if(sh.size(0) != 0)
		{
			M = sh.size(1);
		}
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			time, dt,
			P, S, degree, M,
			background.contiguous().data_ptr<float>(),
			W, H,
			temp_means3D.contiguous().data_ptr<float>(),
			temp_sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			temp_features.contiguous().data_ptr<float>(),
			temp_opacity.contiguous().data_ptr<float>(),
			temp_scales.contiguous().data_ptr<float>(),
			scale_modifier,
			temp_rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			shShaderAddresses.contiguous().data_ptr<int64_t>(),	
			splatShaderAddresses.contiguous().data_ptr<int64_t>(),
			viewmatrix.contiguous().data_ptr<float>(),
			viewmatrix_inv.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			projmatrix_inv.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			cx,
			cy,
			prefiltered,
			computer_pseudo_normal,
			d_textureManager,
			postProcessingPasses,
			out_color.contiguous().data_ptr<float>(),
			out_opacity.contiguous().data_ptr<float>(),
			out_depth.contiguous().data_ptr<float>(),
			out_stencil.contiguous().data_ptr<float>(),
			out_feature.contiguous().data_ptr<float>(),
			out_shader_color.contiguous().data_ptr<float>(),
			out_normal.contiguous().data_ptr<float>(),
			out_surface_xyz.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug);
	}
	char* img_ptr = reinterpret_cast<char*>(imgBuffer.contiguous().data_ptr());
	CudaRasterizer::ImageState imgState = CudaRasterizer::ImageState::fromChunk(img_ptr, H*W);

	torch::Tensor n_contrib = torch::from_blob(imgState.n_contrib, {H, W, 1}, int_opts);
	return std::make_tuple(rendered, n_contrib, out_color, out_opacity, out_depth, out_stencil, out_feature, out_shader_color, out_normal, out_surface_xyz, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& features,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_opacity,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_feature,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool backward_geometry,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int S = features.size(1);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dfeatures = torch::zeros({P, S}, features.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::backward(P, S, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H,
	  means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  features.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_opacity.contiguous().data_ptr<float>(),
	  dL_dout_depth.contiguous().data_ptr<float>(),
	  dL_dout_feature.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dfeatures.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  backward_geometry,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dfeatures, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}