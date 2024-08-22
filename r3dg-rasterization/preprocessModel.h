#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> PreprocessModel(torch::Tensor& splatCoordinateTensor);
