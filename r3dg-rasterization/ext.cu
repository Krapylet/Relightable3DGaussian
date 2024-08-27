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

#include <torch/extension.h>
#include "rasterize_points.h"
#include "render_equation.h"
#include "cuda_rasterizer/splatShader.h"
#include "cuda_rasterizer/shShader.h"
#include "preprocessModel.h"
#include "texture.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("render_equation_forward", &RenderEquationForwardCUDA);
  m.def("render_equation_forward_complex", &RenderEquationForwardCUDA_complex);
  m.def("render_equation_backward", &RenderEquationBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("GetSplatShaderAddressMap", &SplatShader::GetSplatShaderAddressMap);
  m.def("GetShShaderAddressMap", &ShShader::GetShShaderAddressMap);
  m.def("PreprocessModel", &PreprocessModel);
  m.def("decodeTextureMode", &Texture::decodeTextureMode);
  m.def("encodeTextureMode", &Texture::encodeTextureMode);
}