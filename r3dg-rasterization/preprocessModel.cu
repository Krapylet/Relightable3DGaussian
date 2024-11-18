#pragma once

#include "preprocessModel.h"
#include "cuda_rasterizer/ShShader.h"
#include "cuda_rasterizer/splatShader.h"
#include "cuda_rasterizer/auxiliary.h"
#include "shaderManager.h"
#include <cooperative_groups.h>
#include <inttypes.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace cg = cooperative_groups;

__global__ void SelectShadersCUDA(
    glm::vec3* splatCoordinates,
    ShaderManager* shShaderManager,
    ShaderManager* splatShaderManager,
    int splatCount,
    int64_t* shShaderIndexes,
    int64_t* splatShadersIndexes)
{
    // calculate index for the spalt.
    auto idx = cg::this_grid().thread_rank();
    if (idx >= splatCount)
        return;
    auto pos = splatCoordinates[idx];

    shShaderIndexes[idx] = shShaderManager->GetIndexOfShader("ShDefault");
    splatShadersIndexes[idx] = splatShaderManager->GetIndexOfShader("crackNoRecon");
}


void SelectShaders(
    glm::vec3* splatCoordinates,
    ShaderManager* shShaderManager,
    ShaderManager* splatShaderManager,
    int splatCount,
    int64_t* shShaderIndexes,
    int64_t* splatShaderIndexes)
{
    SelectShadersCUDA<<<(splatCount + 255) / 256, 256>>>(
        splatCoordinates,
        shShaderManager,
        splatShaderManager,
        splatCount,
        shShaderIndexes,
        splatShaderIndexes
        );
}

// Currently only sorts on a single thread, so this is extremely slow. Should be optimized, but only runs when the engine starts, so it hasn't been a priority yet.
__global__ void SortShadersCUDA(
    ShaderManager* shShaderManager,
    ShaderManager* splatShaderManager,
    int splatCount, 
    int64_t* shShaderIndexes, 
    int64_t* splatShaderIndexes)
{
    // Count how many times each shader is used
    for (size_t i = 0; i < splatCount; i++)
    {
        int shShader = shShaderIndexes[i];
        int splatShader = splatShaderIndexes[i];

        shShaderManager->d_shaderInstanceCount[shShader]++;
        splatShaderManager->d_shaderInstanceCount[splatShader]++;
    }

    // allocate new memory for storing 3D gaussians indexes by which shader they use    
    int shShaderCount = *shShaderManager->d_shaderCount;
    for (size_t i = 0; i < shShaderCount; i++)
    {
        auto shaderAssociationList = (int*)malloc(shShaderManager->d_shaderInstanceCount[i] * sizeof(int));
        shShaderManager->d_d_shaderAssociationMap[i] = shaderAssociationList;
    }
    
    int splatShaderCount = *splatShaderManager->d_shaderCount;
    for (size_t i = 0; i < splatShaderCount; i++)
    {
        int* shaderAssociationList = (int*)malloc(splatShaderManager->d_shaderInstanceCount[i] * sizeof(int));
        splatShaderManager->d_d_shaderAssociationMap[i] = shaderAssociationList;
    }

    // For each shader, fill out their array with the indexes of the Gaussians that use it.
    int* shShaderInstanceIndexes = (int*)malloc(shShaderCount * sizeof(int));
    int* splatShaderInstanceIndexes = (int*)malloc(splatShaderCount * sizeof(int));
    for (size_t i = 0; i < splatCount; i++)
    {
        // For the current 3D gaussian...
        // Get the type of shader used 
        int shShaderIdx = shShaderIndexes[i];
        int splatShaderIdx = splatShaderIndexes[i];

        // Get how many 3D Gaussians have been assigned to that shader so far
        int shShaderInstanceIdx = shShaderInstanceIndexes[shShaderIdx];
        int splatShaderInstanceIdx = splatShaderInstanceIndexes[splatShaderIdx];

        // Assign the current 3D Gaussian
        shShaderManager->d_d_shaderAssociationMap[shShaderIdx][shShaderInstanceIdx] = i;
        splatShaderManager->d_d_shaderAssociationMap[splatShaderIdx][splatShaderInstanceIdx] = i;

        // increase instance indexes:
        shShaderInstanceIndexes[shShaderIdx]++;
        splatShaderInstanceIndexes[splatShaderIdx]++;
    }
}

void SortShaders(
    ShaderManager* shShaderManager,
    ShaderManager* splatShaderManager,
    int splatCount, 
    int64_t* shShaderIndexes, 
    int64_t* splatShaderIndexes)
{
    SortShadersCUDA<<<1,1>>>(
        shShaderManager,
        splatShaderManager,
        splatCount,
        shShaderIndexes,
        splatShaderIndexes
    );
}


// Figures out which shader each splat should be using and writes it the splatShaders tensor. Currently we only assign shaders
// based on splat coordinates in order to facilitate testing. Ideally, this entire step should be preauthored and worked directly into the splat file or something.
std::tuple<int64_t, int64_t> PreprocessModel(torch::Tensor& splatCoordinateTensor)
{
    // Read input coordinate tensor as an array
    glm::vec3* splatCoordinates = (glm::vec3*)splatCoordinateTensor.contiguous().data_ptr<float>();

    // Initialize output tensors
    int splatCount = splatCoordinateTensor.size(0);
    c10::TensorOptions options = torch::dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor d_shShaderIndexes = torch::empty({splatCount}, options);
    torch::Tensor d_splatShaderIndexes = torch::empty({splatCount}, options);

    // Create shader managers
    ShaderManager* h_shShaderManager = new ShaderManager(ShShader::GetShShaderAddressMap());
    ShaderManager* h_splatShaderManager = new ShaderManager(SplatShader::GetSplatShaderAddressMap());
    
    // Copying managers to device
    ShaderManager* d_shShaderManager;
    ShaderManager* d_splatShaderManager;
    cudaMalloc(&d_shShaderManager, sizeof(ShaderManager));
    cudaMalloc(&d_splatShaderManager, sizeof(ShaderManager));
    cudaMemcpy(d_shShaderManager, h_shShaderManager, sizeof(ShaderManager), cudaMemcpyHostToDevice);
    cudaMemcpy(d_splatShaderManager, h_splatShaderManager, sizeof(ShaderManager), cudaMemcpyHostToDevice);
    
    // Run the address appending on the GPU   
	CHECK_CUDA(SelectShaders(
        splatCoordinates,
        d_shShaderManager,
        d_splatShaderManager,
        splatCount,
        d_shShaderIndexes.contiguous().mutable_data_ptr<int64_t>(),
        d_splatShaderIndexes.contiguous().mutable_data_ptr<int64_t>()
        ), true)

    CHECK_CUDA(SortShaders(
        d_shShaderManager,
        d_splatShaderManager,
        splatCount,
        d_shShaderIndexes.contiguous().mutable_data_ptr<int64_t>(),
        d_splatShaderIndexes.contiguous().mutable_data_ptr<int64_t>()
    ), true)

    // Copying managers back to host
    cudaMemcpy(h_shShaderManager, d_shShaderManager, sizeof(ShaderManager), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_splatShaderManager, d_splatShaderManager, sizeof(ShaderManager), cudaMemcpyDeviceToHost);

    // Finally, copy the maps to host.
    int shShaderCount = h_shShaderManager->h_shaderCount;
    int splatShaderCount = h_splatShaderManager->h_shaderCount;
    cudaMemcpy(h_shShaderManager->h_d_shaderAssociationMap, h_shShaderManager->d_d_shaderAssociationMap, shShaderCount * sizeof(int*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_splatShaderManager->h_d_shaderAssociationMap, h_splatShaderManager->d_d_shaderAssociationMap, splatShaderCount * sizeof(int*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shShaderManager->h_shaderInstanceCount, h_shShaderManager->d_shaderInstanceCount, shShaderCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_splatShaderManager->h_shaderInstanceCount, h_splatShaderManager->d_shaderInstanceCount, splatShaderCount * sizeof(int), cudaMemcpyDeviceToHost);
    
    // delete device managers
    cudaFree(d_shShaderManager);
    cudaFree(d_splatShaderManager);
    
    return std::make_tuple((int64_t)h_shShaderManager, (int64_t)h_splatShaderManager);
}