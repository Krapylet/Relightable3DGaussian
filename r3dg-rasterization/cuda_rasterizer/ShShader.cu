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
    __device__ ShShaderParams::ShShaderParams(PackedShShaderParams p, int idx):
        scale_modifier(p.scale_modifier),
		grid(p.grid),
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix(p.projmatrix),
		projmatrix_inv(p.projmatrix_inv),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),
		W(p.W), H(p.H),
		focal_x(p.focal_x), focal_y(p.focal_y),
		tan_fovx(p.tan_fovx), tan_fovy(tan_fovy),
        deg(p.deg), max_coeffs(p.max_coeffs),

		//input/output   -   contains values when the method is called that can be changed.
		position(p.positions + idx),
		scale(p.scales + idx),
		rotation(p.rotations + idx),
		opacity(p.opacities + idx),
		sh(p.shs + idx * p.max_coeffs) // could also be calculated as idx + (p.deg + 1)^2
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ static void DefaultShShaderCUDA(ShShaderParams p)
    {
        // Set output color
        //*p.out_color = (*p.color_SH);
    }

    __device__ static void ExponentialPositionShaderCUDA(ShShaderParams p)
    {
        // multiply sh position and scale by y coordinate
        float posX = abs((*p.position).x);
        float posY = abs((*p.position).y);
        float posZ = abs((*p.position).z);
        float dist = (*p.position).length();
        
        *p.scale = glm::vec3((*p.scale).x * posY, (*p.scale).y * 2, (*p.scale).z) * posY;
        *p.position = glm::vec3((*p.position).x * posY, (*p.position).y * 2, (*p.position).z) * posY;
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    __device__ const ShShader defaultShader = &DefaultShShaderCUDA;
    __device__ const ShShader expPosShader = &ExponentialPositionShaderCUDA;

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
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderMap["Default"] = (int64_t)h_defaultShader;

        ShShader::ShShader h_exponentialPositionShader;
        cudaMemcpyFromSymbol(&h_exponentialPositionShader, expPosShader, shaderMemorySize);
        shaderMap["ExpPos"] = (int64_t)h_exponentialPositionShader;

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
        shaders[idx](params);

        //DefaultShShaderCUDA(params);
    }

}