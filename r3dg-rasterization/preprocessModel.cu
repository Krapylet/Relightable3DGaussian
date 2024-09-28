#pragma once

#include "preprocessModel.h"
#include "cuda_rasterizer/ShShader.h"
#include "cuda_rasterizer/splatShader.h"
#include "cuda_rasterizer/auxiliary.h"
#include "utils/indirectMap.h"
#include <cooperative_groups.h>
#include <inttypes.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace cg = cooperative_groups;

__global__ void AppendShaderIndexes(
    glm::vec3* splatCoordinates,
    IndirectMap<char*, ShShader::ShShader>* shShaderAddressMap,
    IndirectMap<char*, SplatShader::SplatShader>* splatShaderAddressMap,
    int splatCount,
    ShShader::ShShader* out_shShaders,
    SplatShader::SplatShader* out_splatShaders)
{
    // calculate index for the spalt.
    auto idx = cg::this_grid().thread_rank();
    if (idx >= splatCount)
        return;

    // Determine which shader should be used for the splat.
    char* shShaderName = "Default";
    char* splatShaderName = "Default";

    if (splatCoordinates[idx].x > 0){
        shShaderName = "Diffuse";
    }

    // Assign 
    out_shShaders[idx] = shShaderAddressMap->Get(shShaderName);
    out_splatShaders[idx] = splatShaderAddressMap->Get(splatShaderName);
}


// Figures out which shader each splat should be using and writes it the splatShaders tensor. Currently we only assign shaders
// based on splat coordinates in order to facilitate testing. Ideally, this entire step should be preauthored and worked directly into the splat file or something.
std::tuple<int64_t, int64_t> PreprocessModel(torch::Tensor& splatCoordinateTensor)
{
    // Load shader addresses. Allocates memory
    IndirectMap<char*, ShShader::ShShader>* shShaderAddressMap = ShShader::GetShShaderAddressMap();
    IndirectMap<char*, SplatShader::SplatShader>* splatShaderAddressMap = SplatShader::GetSplatShaderAddressMap();

    // Read input coordinate tensor as an array
    glm::vec3* splatCoordinates = (glm::vec3*)splatCoordinateTensor.contiguous().data_ptr<float>();

    // Initialize output arrays
    int splatCount = splatCoordinateTensor.size(0);
    ShShader::ShShader* shShaders;
    cudaMalloc(shShaders, splatCount * sizeof(ShShader::ShShader));

    SplatShader::SplatShader* splatShaders;
    cudaMalloc(splatShaders, splatCount * sizeof(SplatShader::SplatShader));

    // Run the address appending on the GPU   
	AppendShaderIndexes<<<(splatCount + 255) / 256, 256>>>(
        splatCoordinates,
        shShaderAddressMap,
        splatShaderAddressMap,
        splatCount,
        shShaders,
        splatShaders
        );

    // Free memory allocated by the address getter functions.
    delete shShaderAddressMap;
    delete splatShaderAddressMap;
    
    return std::make_tuple((int64_t)shShaders, (int64_t)splatShaders);
}