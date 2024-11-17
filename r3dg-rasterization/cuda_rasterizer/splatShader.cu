#include "splatShader.h"
#include "config.h"
#include <cooperative_groups.h>
#include "../utils/shaderUtils.h"
#include "auxiliary.h"


namespace cg = cooperative_groups;

namespace SplatShader
{
    __device__ SplatShaderConstantInputs::SplatShaderConstantInputs(PackedSplatShaderParams p, int idx):
        W(p.W),
        H(p.H),
        time(p.time), dt(p.dt),
		position(p.positions[idx]),	
		screen_position(p.screen_positions[idx]),
        depth_tex(p.depth_tex),
        stencil_tex(p.stencil_tex),
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix (p.projmatrix),
		projmatrix_inv (p.projmatrix_inv),
		focal_x (p.focal_x),
        focal_y (p.focal_y),
		tan_fovx (p.tan_fovx), 
        tan_fovy (p.tan_fovy),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),

		// pr. frame texture information
		splat_depth (p.splat_depths[idx]),		
		conic ((glm::vec3)p.conic_opacity[idx]), 
        mean_pixel_idx (p.W * floorf(screen_position.y) + floorf(screen_position.x)),
        color_SH (p.colors_SH + idx),

        // Texture information
        d_textureManager(p.d_textureManager)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ SplatShaderModifiableInputs::SplatShaderModifiableInputs(PackedSplatShaderParams p, int idx):
        // Precomputed pr. splat 'texture' information from the neilf pbr decomposition
        roughness (p.features + idx * p.S + 0),
		metallic (p.features + idx * p.S + 1),
		incident_visibility (p.features + idx * p.S + 2),
		color_brdf ((glm::vec3*)p.features + idx * p.S + 3),
		normal ((glm::vec3*)p.features + idx * p.S + 6),
		color_base ((glm::vec3*)p.features + idx * p.S + 9),
		incident_light ((glm::vec3*)p.features + idx * p.S + 12),
		local_incident_light ((glm::vec3*)p.features + idx * p.S + 15),
		global_incident_light ((glm::vec3*)p.features + idx * p.S + 18),

		// pr. splat information
		opacity (((float*)p.conic_opacity) + idx * 4 + 3),  // Opacity works a bit funky because how splats are blended. It is better to multiply this paramter by something rather than setting it to specific values.
        stencil_val (p.stencils + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ SplatShaderOutputs::SplatShaderOutputs(PackedSplatShaderParams p, int idx):
        out_color (p.out_colors + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }



    __device__ static void DefaultSplatShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        // Set output color
        *out.out_color = (*in.color_SH);
    }

    // A naive shader for hgihlighting edges on model.
    __device__ static void NaiveOutlineShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = in.camera_position - in.position;
        float angle = 1 - glm::abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(*io.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        // Set output color
        *out.out_color = (*in.color_SH) * opacity;
    }

    __device__ static void WireframeShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = in.camera_position - in.position;
        float angle = 1 - glm::abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(*io.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        float rColor = fmodf(in.time / 5000, 1.0);
        // Set output color
        *out.out_color = glm::vec3(rColor, 1 - opacity,  1 - opacity);
    }

    // Makes the object fade in and out.
    // Written in splat shader because this is where we have best access to colors.
    __device__ static void DissolveShader(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){
        cudaTextureObject_t grainyTexture = in.d_textureManager->GetTexture("Grid");

        // Grab the opacity from a mask texture
        float maskSample_xy = tex2D<float4>(grainyTexture, in.position.x, in.position.y).x;
        float maskSample_xz = tex2D<float4>(grainyTexture, in.position.x, in.position.z).x;
        float maskSample_yz = tex2D<float4>(grainyTexture, in.position.y, in.position.z).x;
        
        // combine masking from the 3 planes to create a 3d mask.
        float maskSample = maskSample_xy * maskSample_xz * maskSample_yz;

        // goes back and forth between 0 and 1 over time
        float opacityPercent = (cosf(in.time/4000) + 1)/2;

        // Offset the opacity by the mask
        float opacity = __saturatef((1 + maskSample) * opacityPercent);

        // Ease in and out of transparency with a quint easing.
        float easedOpacity = opacity < 0.5 ? 16.0 * powf(opacity, 5) : 1 - powf(-2 * opacity + 2, 5) / 2;

        float originalOpacity = *io.opacity;

        // Opacity output
        *io.opacity = easedOpacity * originalOpacity;

        // We want the colors to turn progressively more bright blue as they turn transparen
        glm::vec3 targetfadeColor = glm::vec3(0.6,0.9,1);
        float fadeColorPercent = __saturatef(1-opacity -0.3);// ;
        float fadeColorEasing = fadeColorPercent < 0.5 ? 16.0 * powf(fadeColorPercent + 0.1, 5) : 1 - powf(-2 * fadeColorPercent + 2, 5) / 2;

        // mix degree the fade color into the base color
        *out.out_color = glm::mix(*in.color_SH, targetfadeColor, fadeColorEasing);
    }

    __device__ static void CrackShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        cudaTextureObject_t crackTex = in.d_textureManager->GetTexture("Depth cracks");
        // Rescale UVs
        // Currently we just project directly downwards, but the projection can be rotated and pivoted to anywhere around the model.
        float texScale = 2;
        float u = in.position.x/texScale - 0.5;
        float v = in.position.y/texScale - 0.5;
        float crackTexDepth = 1 - tex2D<float4>(crackTex, u, v).x;

        // First, delete all splats inside the crack by making them completely transparent. 
        float maxCrackDepth = 2;
        float projectionHeight = 2;
        float crackHeight = projectionHeight - crackTexDepth * maxCrackDepth;
        float splatHeight = in.position.z;

        bool crackReachesSplat = crackHeight < splatHeight;
        *io.opacity = crackReachesSplat ? 0 : *io.opacity;

        // Figure out which splats are beneath the surface of the model
        float depthTolerance = 0.2f; // Increasing this value causes more splats to be considered internal.
        float distToSurface = in.splat_depth - in.depth_tex[in.mean_pixel_idx] - depthTolerance;
        bool splatIsInsideModel = distToSurface < 0;
        
        // Splats that are both close to the deleted splats AND inside the model gets a completely new color.
        float internalColorReach = 0.1f;
        float maxPrimaryColorHeight = projectionHeight - (crackTexDepth + internalColorReach) * maxCrackDepth;
        bool SplatIsInCrackColorReach = splatHeight > maxPrimaryColorHeight;
        bool shouldUseInternalColor = splatIsInsideModel && SplatIsInCrackColorReach;
        bool internalColorPercent = __saturatef(distToSurface * 10);
        glm::vec3 internalColor = glm::mix(*io.color_base, glm::vec3(0.5f, 0.5f, 0), internalColorPercent );

        // Splats that are not inside the model get discolored a bit based on their distance ot the crack.
        float discolorReach = 0.1f;
        float maxDiscolorHeight = maxPrimaryColorHeight - discolorReach * maxCrackDepth;
        float discolorPercent =  __saturatef((splatHeight - maxDiscolorHeight) / (discolorReach + internalColorReach));
        glm::vec3 externalColor = glm::mix(*in.color_SH, internalColor, discolorPercent);

        // Sample the texture a couple of times more to calculate the normal of the slope inside the crack
        float uvOffset = 0.01f; // in UV coords [0-1]
        float resampleDist = uvOffset * texScale;
        float crackTexDepth_North = 1 - tex2D<float4>(crackTex, u, v + uvOffset).x;
        float crackTexDepth_South = 1 - tex2D<float4>(crackTex, u, v - uvOffset).x;
        float crackTexDepth_East = 1 - tex2D<float4>(crackTex, u + uvOffset, v).x;
        float crackTexDepth_West = 1 - tex2D<float4>(crackTex, u - uvOffset, v).x;

        glm::vec3 tanget = glm::normalize(glm::vec3(0, resampleDist, crackTexDepth_North - crackTexDepth_South));
        glm::vec3 bitanget = glm::normalize(glm::vec3(resampleDist, 0, crackTexDepth_East - crackTexDepth_West));
        glm::vec3 crackNormal = glm::cross(tanget, bitanget);

        // Use the slope inside the crack to apply very simple shadow to the internal color.
        // Light is shined down directly from above.
        //glm::vec3 viewDir = glm::vec3(p.viewmatrix[9], p.viewmatrix[10], p.viewmatrix[11]); 
        glm::vec3 lightDir = glm::vec3(0, 0, -1);
        float ambientLight = 0.5f;
        //internalColor *= __saturatef(glm::dot(lightDir, crackNormal)/2 + ambientLight);

        glm::vec3 finalColor = internalColor * (float)shouldUseInternalColor + externalColor * (float)!shouldUseInternalColor;
        *io.opacity += 0.1f * (float)shouldUseInternalColor * (float)!crackReachesSplat; // Increase opacity of internal splats  
        *out.out_color = finalColor;
    }

    __device__ static void WriteToStencilCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){
        *io.stencil_val = 1;
        *out.out_color = *in.color_SH;
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    __device__ const SplatShader defaultShader = &DefaultSplatShaderCUDA;
    __device__ const SplatShader naiveOutlineShader = &NaiveOutlineShaderCUDA;
    __device__ const SplatShader wireframeShader = &WireframeShaderCUDA;
    __device__ const SplatShader dissolveShader = &DissolveShader;
    __device__ const SplatShader crackShader = &CrackShaderCUDA;
    __device__ const SplatShader stencilShader = &WriteToStencilCUDA;


    std::map<std::string, int64_t> GetSplatShaderAddressMap(){
        // we cast pointers to numbers since most pointers aren't supported by pybind
        // Device function pointers seem to be 8 bytes long (at least on the devlopment machine with a GTX 2080 and when compiling to 64bit mode)
        // there doesn't seem to be a problem casting them to int64's though.

        std::map<std::string, int64_t> shaderMap;
        size_t shaderMemorySize = sizeof(SplatShader);
        
        // Copy device shader pointers to host map
        SplatShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderMap["SplatDefault"] = (int64_t)h_defaultShader;

        SplatShader h_outlineShader;
        cudaMemcpyFromSymbol(&h_outlineShader, naiveOutlineShader, shaderMemorySize);
        shaderMap["NaiveOutline"] = (int64_t)h_outlineShader;

        SplatShader h_wireframeShader;
        cudaMemcpyFromSymbol(&h_wireframeShader, wireframeShader, shaderMemorySize);
        shaderMap["WireframeShader"] = (int64_t)h_wireframeShader;

        SplatShader h_dissolveShader;
        cudaMemcpyFromSymbol(&h_dissolveShader, dissolveShader, shaderMemorySize);
        shaderMap["dissolveShader"] = (int64_t)h_dissolveShader;

        SplatShader h_crackShader;
        cudaMemcpyFromSymbol(&h_crackShader, crackShader, shaderMemorySize);
        shaderMap["Crack"] = (int64_t)h_crackShader;

        SplatShader h_stencilShader;
        cudaMemcpyFromSymbol(&h_stencilShader, stencilShader, shaderMemorySize);
        shaderMap["Stencil"] = (int64_t)h_stencilShader;

        return shaderMap;
    }

    // ALLOCATES THE RETURN ARRAY. REMEMBER TO FREE AFTER USE.
    // Returns an array in device memory containing addresses to device shader functions.
    int64_t* GetSplatShaderAddressArray(){
        // Array is assembled on CPU before being sent to device. Addresses themselves are in device space.
        int shaderCount = 6;
        int64_t* h_shaderArray = new int64_t[shaderCount];
        size_t shaderMemorySize = sizeof(SplatShader);
 
        SplatShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        h_shaderArray[0] = (int64_t)h_defaultShader;

        SplatShader h_outlineShader;
        cudaMemcpyFromSymbol(&h_outlineShader, naiveOutlineShader, shaderMemorySize);
        h_shaderArray[1] = (int64_t)h_outlineShader;

        SplatShader h_wireframeShader;
        cudaMemcpyFromSymbol(&h_wireframeShader, wireframeShader, shaderMemorySize);
        h_shaderArray[2] = (int64_t)h_wireframeShader;

        SplatShader h_dissolveShader;
        cudaMemcpyFromSymbol(&h_dissolveShader, dissolveShader, shaderMemorySize);
        h_shaderArray[3] = (int64_t)h_dissolveShader;
        
        SplatShader h_crackShader;
        cudaMemcpyFromSymbol(&h_crackShader, crackShader, shaderMemorySize);
        h_shaderArray[4] = (int64_t)h_crackShader;

        SplatShader h_stencilShader;
        cudaMemcpyFromSymbol(&h_stencilShader, stencilShader, shaderMemorySize);
        h_shaderArray[5] = (int64_t)h_stencilShader;

        // copy the host array to device
        int64_t* d_shaderArray;
        cudaMalloc(&d_shaderArray, sizeof(int64_t)*shaderCount);
        cudaMemcpy(d_shaderArray, h_shaderArray, shaderMemorySize * shaderCount, cudaMemcpyDefault);

        // Delete temporary host array.
        delete[] h_shaderArray;
        return d_shaderArray;
    }


    __global__ void ExecuteShader(SplatShader* shaders, PackedSplatShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        if (idx >= packedParams.P)
            return;

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        SplatShaderConstantInputs in(packedParams, idx);
        SplatShaderModifiableInputs io(packedParams, idx);
        SplatShaderOutputs out(packedParams, idx);

        // Debug print statement for seeing what's going on inside shader kernels.
        //if (idx == 1)
            //printf("Position: (%f, %f, %f); Depth:%f; Pixel Coord: (%f, %f)\n", params.position.x, params.position.y, params.position.z, params.prerendered_depth_buffer[0], params.screen_position.x, params.screen_position.y);

        // No need to dereference the shader function pointer.
        shaders[idx](in, io, out);
    }

}

