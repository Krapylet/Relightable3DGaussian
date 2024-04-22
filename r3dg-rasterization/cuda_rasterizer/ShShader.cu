#include "ShShader.h"
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

    }

    ///// Assign all the shaders to their short handles.
    __device__ ShShader defaultShader = &DefaultShShaderCUDA;

    __global__ void ExecuteShader(ShShader shader, PackedShShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        if (idx >= packedParams.splatsInShader)
            return;
        idx += packedParams.shaderStartingOffset;

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        ShShaderParams params(packedParams, idx);

        // No need to dereference the shader function pointer.
        shader(params);
    }

}

