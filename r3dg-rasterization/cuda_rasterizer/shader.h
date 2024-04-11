#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#define GLM_FORCE_CUDA


namespace CudaShader
{
    typedef void (*shader)(
        int W, int H,
		int P,
		const float* orig_points,
		float2* points_xy_image,
		const float* viewmatrix,
		const float* viewmatrix_inv,
		const float* projmatrix,
		const float* projmatrix_inv,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		float* depths,
		float* colors,
		float4* conic_opacity,
		int S,
		const float* features,
		float* out_color);
}
