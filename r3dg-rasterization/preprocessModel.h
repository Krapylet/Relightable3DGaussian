#pragma once

#include "utils/includeTorch.cu"

std::tuple<torch::Tensor, torch::Tensor> PreprocessModel(torch::Tensor& splatCoordinateTensor);
