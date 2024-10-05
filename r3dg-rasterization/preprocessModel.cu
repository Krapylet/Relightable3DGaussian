#pragma once

#include "preprocessModel.h"
#include "cuda_rasterizer/ShShader.h"
#include "cuda_rasterizer/splatShader.h"
#include "cuda_rasterizer/auxiliary.h"
#include <cooperative_groups.h>
#include <inttypes.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace cg = cooperative_groups;

__global__ void AppendShaderIndexesCUDA(
    glm::vec3* splatCoordinates,
    int64_t* shShaderAddressArray,
    int64_t* splatShaderAddressArray,
    int splatCount,
    int64_t* out_shShaderAddresses,
    int64_t* out_splatShadersAddresses)
{
    // calculate index for the spalt.
    auto idx = cg::this_grid().thread_rank();
    if (idx >= splatCount)
        return;

    int shShaderAddressIndex = 3;
    int splatShaderAddressIndex = 0;

    out_shShaderAddresses[idx] = shShaderAddressArray[shShaderAddressIndex];
    out_splatShadersAddresses[idx] = splatShaderAddressArray[splatShaderAddressIndex];
}


void AppendShaderIndexes(
    glm::vec3* splatCoordinates,
    int64_t* shShaderAddressArray,
    int64_t* splatShaderAddressArray,
    int splatCount,
    int64_t* out_shShaderAddresses,
    int64_t* out_splatShaderAddresses)
{
    AppendShaderIndexesCUDA<<<(splatCount + 255) / 256, 256>>>(
        splatCoordinates,
        shShaderAddressArray,
        splatShaderAddressArray,
        splatCount,
        out_shShaderAddresses,
        out_splatShaderAddresses
        );
}

// Figures out which shader each splat should be using and writes it the splatShaders tensor. Currently we only assign shaders
// based on splat coordinates in order to facilitate testing. Ideally, this entire step should be preauthored and worked directly into the splat file or something.
std::tuple<torch::Tensor, torch::Tensor> PreprocessModel(torch::Tensor& splatCoordinateTensor)
{
    // Load shader addresses. Allocates memory
    int64_t* shShaderAddressArray = ShShader::GetShShaderAddressArray();
    int64_t* splatShaderAddressArray = SplatShader::GetSplatShaderAddressArray();

    // Read input coordinate tensor as an array
    glm::vec3* splatCoordinates = (glm::vec3*)splatCoordinateTensor.contiguous().data_ptr<float>();

    // Initialize output tensors
    int splatCount = splatCoordinateTensor.size(0);
    c10::TensorOptions options = torch::dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor d_out_ShShaderAddresses = torch::empty({splatCount}, options);
    torch::Tensor d_out_splatShaderAddresses = torch::empty({splatCount}, options);

    // Run the address appending on the GPU   
	CHECK_CUDA(AppendShaderIndexes(
        splatCoordinates,
        shShaderAddressArray,
        splatShaderAddressArray,
        splatCount,
        d_out_ShShaderAddresses.contiguous().mutable_data_ptr<int64_t>(),
        d_out_splatShaderAddresses.contiguous().mutable_data_ptr<int64_t>()
        ), true)

    std::cout << "Appending done. Cloning data to host.\n" << std::endl;

    torch::Tensor h_out_ShShaderAddresses = d_out_ShShaderAddresses.contiguous().cpu();
    torch::Tensor h_out_splatShaderAddresses = d_out_splatShaderAddresses.contiguous().cpu();

    // Free memory allocated by the address getter functions.
    cudaFree(shShaderAddressArray);
    cudaFree(splatShaderAddressArray);
    
    return std::make_tuple(h_out_ShShaderAddresses, h_out_splatShaderAddresses);
}