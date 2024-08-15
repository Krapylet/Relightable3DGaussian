#pragma once

#include <torch/extension.h>
#include <string>
#include <map>

#ifndef GLM_FORCE_CUDA
	#define GLM_FORCE_CUDA
#endif

#include <glm/glm.hpp>

std::tuple<torch::Tensor, torch::Tensor> PreprocessModel(torch::Tensor& splatCoordinateTensor);
