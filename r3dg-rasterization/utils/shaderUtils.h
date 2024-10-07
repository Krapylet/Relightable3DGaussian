#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

#ifndef GLM_FORCE_CUDA
    #define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

__device__ glm::vec3 RgbToHsv(glm::vec3 rgb);
__device__ glm::vec3 HsvToRgb(glm::vec3 hsv);