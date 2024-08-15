#include "cuda_rasterizer/ShShader.h"
#include "cuda_rasterizer/splatShader.h"
#include <torch/extension.h>
#include <cooperative_groups.h>
#include "cuda_rasterizer/auxiliary.h"
#include <inttypes.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace cg = cooperative_groups;

__global__ void AppendShaderIndexesCUDA(
    glm::vec3* splatCoordinates,
    int64_t* shShaderAdressArray,
    int64_t* splatShaderAdressArray,
    int splatCount,
    int64_t* out_shShaderAdresses,
    int64_t* out_splatShadersAdresses)
{
    // calculate index for the spalt.
    auto idx = cg::this_grid().thread_rank();
    if (idx >= 10)
        return;

    int shShaderAdressIndex = 0;
    int splatShaderAdressIndex = 0;

    // Determine which shader should be used for the splat.
    // Ideally assigned shaders sould be written direcly in the object file so we know this when we load the model in.
    // also, there's no propper naming here unlike in the regular append_shader_addresses() python function, so it's only a placeholder for testing initialization speed.
    if (splatCoordinates[idx].x > 0){
        shShaderAdressIndex = 0; //"Default";
    }
    else{
        shShaderAdressIndex = 1 ;//"ExpPos";
    }

    if (splatCoordinates[idx].y > 0){
        splatShaderAdressIndex = 0;// "Default";
    }
    else{
        splatShaderAdressIndex = 2;//"WireframeShader";
    }

    //int64_t addressTest = shShaderAdressArray[shShaderAdressIndex];
    //int64_t outputTest = out_shShaderAdresses[idx];

    //printf("Adress test for idx" PRIx64 ": %llu\n", addressTest, idx);
    //printf("output test for idx %d: %d\n", outputTest, idx);
    //out_shShaderAdresses[idx] = shShaderAdressArray[shShaderAdressIndex];
    //out_splatShadersAdresses[idx] = splatShaderAdressArray[splatShaderAdressIndex];
}


void AppendShaderIndexes(
    glm::vec3* splatCoordinates,
    int64_t* shShaderAdressArray,
    int64_t* splatShaderAdressArray,
    int splatCount,
    int64_t* out_shShaderAdresses,
    int64_t* out_splatShaderAdresses)
{
    AppendShaderIndexesCUDA<<<(splatCount + 255) / 256, 256>>>(
        splatCoordinates,
        shShaderAdressArray,
        splatShaderAdressArray,
        splatCount,
        out_shShaderAdresses,
        out_splatShaderAdresses
        );
}

// Figures out which shader each splat should be using and writes it the splatShaders tensor. Currently we only assign shaders
// based on splat coordinates in order to facilitate testing. Ideally, this entire step should be preauthored and worked directly into the splat file or something.
std::tuple<torch::Tensor, torch::Tensor> PreprocessModel(torch::Tensor& splatCoordinateTensor)
{
    // Load shader adresses
    int64_t* shShaderAdressArray = ShShader::GetShShaderAddressArray();
    int64_t* splatShaderAdressArray = SplatShader::GetSplatShaderAddressArray();

    // Read input tensors as arays
    glm::vec3* splatCoordinates = (glm::vec3*)splatCoordinateTensor.contiguous().data_ptr<float>();

    // Initialize output tensors
    int splatCount = splatCoordinateTensor.size(0);
    auto options = splatCoordinateTensor.options().dtype(torch::kInt64).device(torch::kCUDA); // copy the options from the splatCoordinate tensor, but overwrite the datatype.
    torch::Tensor out_ShShaderAdresses = torch::full({splatCount}, shShaderAdressArray[0], options);
    torch::Tensor out_splatShaderAdresses = torch::full({splatCount}, splatShaderAdressArray[0], options);

    //std::cout << "Current value: " << out_ShShaderAdresses.contiguous().mutable_data_ptr<int64_t>()[0] << std::endl;

    // Run the address appending on the GPU   
	CHECK_CUDA(AppendShaderIndexes(
        splatCoordinates,
        shShaderAdressArray,
        splatShaderAdressArray,
        splatCount,
        out_ShShaderAdresses.contiguous().mutable_data_ptr<int64_t>(),
        out_splatShaderAdresses.contiguous().mutable_data_ptr<int64_t>()
        ), true)

    std::cout << "Appending done" << std::endl;

    // We have to remember the allocated memory of the shader arrays. Ideally we shouldn't need to do this at all, though.
    //delete[] shShaderAdressArray;
    //delete[] splatShaderAdressArray;

    return std::make_tuple(out_ShShaderAdresses, out_splatShaderAdresses);
}