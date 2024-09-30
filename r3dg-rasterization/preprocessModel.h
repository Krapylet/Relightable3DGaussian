#pragma once

#include "utils/includeTorch.cu"

std::tuple<int64_t, int64_t> PreprocessModel(torch::Tensor& splatCoordinateTensor);
