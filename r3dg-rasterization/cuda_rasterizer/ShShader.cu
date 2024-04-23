#include "shShader.h"
#include "config.h"
#include <cooperative_groups.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>


namespace cg = cooperative_groups;

namespace ShShader
{
    __device__ ShShaderParams::ShShaderParams(PackedShShaderParams p, int idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ static void DefaultShShaderCUDA(ShShaderParams p)
    {
        // Set output color
        //*p.out_color = (*p.color_SH);
    }

    std::map<std::string, int64_t> GetShShaderAddressMap(){
        // we cast pointers to numbers since most pointers aren't supported by pybind
        // Device function pointers seem to be 8 bytes long (at least on the devlopment machine with a GTX 2080 and when compiling to 64bit mode)
        // The highest unsigned integer supported by torch, which we use for contigious memory, is 1 byte ints.
        // This means we can either cut the pointer into 8 small ints when we send them back and forth to the python frontend,
        // Or we can try to make our own pybind datatype binding.
        // alternatively, we can try to do our own casting by using bitwise OR to encode the pointer into a signed int64 anyway.

        std::map<std::string, int64_t> shaderMap;
        size_t shaderMemorySize = sizeof(ShShader);
        
        // Copy device shader pointers to host map
        ShShader::ShShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, &DefaultShShaderCUDA, shaderMemorySize);
        shaderMap["Default"] = (int64_t)h_defaultShader;

        return shaderMap;
    }

    __global__ void ExecuteShader(ShShader* shaders, PackedShShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        if (idx >= packedParams.P)
            return;

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        ShShaderParams params(packedParams, idx);

        // No need to dereference the shader function pointer.
        //shaders[idx](params);

        DefaultShShaderCUDA(params);
    }

}