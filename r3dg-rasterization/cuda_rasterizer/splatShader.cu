#include "splatShader.h"
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
		color_brdf ((glm::vec3*)&p.features[idx * p.S + 3]),
		normal ((glm::vec3*)&p.features[idx * p.S + 6]),
		color_base ((glm::vec3*)&p.features[idx * p.S + 9]),
		incident_light ((glm::vec3*)&p.features[idx * p.S + 12]),
		local_incident_light ((glm::vec3*)&p.features[idx * p.S + 15]),
		global_incident_light ((glm::vec3*)&p.features[idx * p.S + 18]),

		// pr. splat information
		opacity (((float*)p.conic_opacity) + idx * 4 + 3),  // Opacity works a bit funky because how splats are blended. It is better to multiply this paramter by something rather than setting it to specific values.
        stencil_val (p.stencils + idx),
        stencil_opacity (p.stencil_opacities + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ SplatShaderOutputs::SplatShaderOutputs(PackedSplatShaderParams p, int idx):
        out_color (p.out_colors + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ void DefaultSplatShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        // Set output color
        *out.out_color = (*in.color_SH);
    }

    // A naive shader for hgihlighting edges on model.
    __device__ void NaiveOutlineShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
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

    __device__ void WireframeShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = in.camera_position - in.position;
        float angle = 1 - glm::abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(*io.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        // Set output color
        *out.out_color = glm::vec3(1 - opacity, 1 - opacity,  1 - opacity);
    }

    // Makes the object fade in and out.
    // Written in splat shader because this is where we have best access to colors.
    __device__ void DissolveShader(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){
        cudaTextureObject_t maskTex = in.d_textureManager->GetTexture("Cracks");

        // Grab the opacity from a mask texture
        float maskSample_xy = tex2D<float4>(maskTex, in.position.x, in.position.y).x;
        float maskSample_xz = tex2D<float4>(maskTex, in.position.x, in.position.z).x;
        float maskSample_yz = tex2D<float4>(maskTex, in.position.y, in.position.z).x;
        
        // combine masking from the 3 planes to create a 3d mask.
        float maskSample = maskSample_xy * maskSample_xz * maskSample_yz;

        // make the mask less gray
        maskSample = __saturatef((maskSample-0.125)*1.5);

        // How often to repeat each second
        float period = 0.1f; 

        // goes back and forth between 0 and 1 over time
        float opacity = (cosf(in.time * period * 4 / (M_1_PI * 2 * 1000)) + 1);

        // Offset the opacity by the mask  
        float maskedOpacity = __saturatef(opacity - (1 - maskSample));

        // Ease in and out of transparency with a quint easing.
        //float easedFade = fadeAmount < 0.5 ? 16.0 * powf(fadeAmount, 5) : 1 - powf(-2 * fadeAmount + 2, 5) / 2;

        // Opacity output
        *io.opacity = *io.opacity * maskedOpacity;

        float colorFading = __saturatef(maskedOpacity*3);
        *io.stencil_val = maskSample;

        // We want the colors to turn progressively more bright blue as they turn transparen
        glm::vec3 targetfadeColor = glm::vec3(0.6,0.9,1);
        *out.out_color = glm::mix(targetfadeColor, *in.color_SH, colorFading);
    }

    __device__ void CrackShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
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
        float depthTolerance = 0.3f; // Increasing this value causes more splats to be considered internal.
        float distToSurface = in.splat_depth - in.depth_tex[in.mean_pixel_idx] + depthTolerance;
        bool splatIsInsideModel = distToSurface > 0;
        
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

        glm::vec3 finalColor = internalColor * (float)shouldUseInternalColor + externalColor * (float)!shouldUseInternalColor;
        *io.opacity += 0.2f * (float)shouldUseInternalColor * (float)!crackReachesSplat; // Increase opacity of internal splats  
        *out.out_color = finalColor;
    }

    __device__ void CrackWithoutReconstructionShaderCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out)
    {
        cudaTextureObject_t crackTex = in.d_textureManager->GetTexture("Bulge");
        //cudaTextureObject_t crackTex = in.d_textureManager->GetTexture("Depth cracks");
        
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
        float originalOpacity = *io.opacity;
        *io.opacity = crackReachesSplat ? 0 : *io.opacity;

        // Figure out which splats are beneath the surface of the model
        float depthTolerance = 0.2f; // Increasing this value causes more splats to be considered internal.
        float depthRelativeToSurface = in.splat_depth - in.depth_tex[in.mean_pixel_idx] + depthTolerance;
        bool splatIsInsideModel = depthRelativeToSurface > 0;
        
        // Splats that are both close to the deleted splats AND inside the model gets a completely new color.
        float internalColorReach = 0.5f * crackTexDepth;
        float maxPrimaryColorHeight = projectionHeight - (crackTexDepth + internalColorReach) * maxCrackDepth;
        bool SplatIsInCrackColorReach = maxPrimaryColorHeight < splatHeight;
        bool shouldUseInternalColor = splatIsInsideModel && SplatIsInCrackColorReach;
        
        *out.out_color = *io.color_base;

        // Write masked out splats to stencil
        *io.stencil_val = crackReachesSplat;
        *io.stencil_opacity = originalOpacity;

        // Write sorrounding splats to seperate stencil.
        float opacityIncrease = 2.;// increase opacity of nearby splats.
        *io.metallic = shouldUseInternalColor;
    }

    __device__ void WriteToStencilCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){
        *io.stencil_val = 1;
        *io.stencil_opacity = *io.opacity;
        *out.out_color = *in.color_SH;
    }

    __device__ void RoughnessOnlyCUDA(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){
        if(in.position.x < 0){
            *io.roughness = 0.25f;
        }
        else{
            *io.roughness = 0.75f;
        }
        
        *io.metallic = 0;
		*io.incident_visibility = 0;
		//*io.color_brdf = glm::vec3(0, 0.5, 1);
		*io.normal = glm::vec3(0);
		*io.color_base = glm::vec3(0);
		*io.incident_light = glm::vec3(0);
		*io.local_incident_light = glm::vec3(0);
		*io.global_incident_light = glm::vec3(0);
        *out.out_color = glm::vec3(0);
    }

    __device__ void QuantizeFlatColors(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){

        glm::vec3 quantizedLight = Quantize(*io.incident_light, 3);
		*out.out_color = *io.color_base;//* quantizedLight;
    }

    __device__ void QuantizeLight(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out){

        glm::vec3 qIntensity = Quantize(*io.incident_light, 3);
        float whiteIntensity = max(qIntensity.r, max(qIntensity.g, qIntensity.b)); // Convert the light to whit efor a simpler picture.

        //*io.incident_light = glm::vec3(whiteIntensity);
        // Write to roguhness instead during testing
        *io.roughness = whiteIntensity;

		*out.out_color = *io.color_base;//* quantizedLight;
    }

    ///// Assign all the shaders to their short handles.
    __device__ SplatShader defaultShader = &DefaultSplatShaderCUDA;
    __device__ SplatShader naiveOutlineShader = &NaiveOutlineShaderCUDA;
    __device__ SplatShader wireframeShader = &WireframeShaderCUDA;
    __device__ SplatShader dissolveShader = &DissolveShader;
    __device__ SplatShader crackShader = &CrackShaderCUDA;
    __device__ SplatShader crackNoReconShader = &CrackWithoutReconstructionShaderCUDA;
    __device__ SplatShader stencilShader = &WriteToStencilCUDA;
    __device__ SplatShader roughnessOnly = &RoughnessOnlyCUDA;
    __device__ SplatShader quantizeFlats = &QuantizeFlatColors;
    __device__ SplatShader quantizeLight = &QuantizeLight;


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
        shaderMap["Wireframe"] = (int64_t)h_wireframeShader;

        SplatShader h_dissolveShader;
        cudaMemcpyFromSymbol(&h_dissolveShader, dissolveShader, shaderMemorySize);
        shaderMap["Dissolve"] = (int64_t)h_dissolveShader;

        SplatShader h_crackShader;
        cudaMemcpyFromSymbol(&h_crackShader, crackShader, shaderMemorySize);
        shaderMap["Crack"] = (int64_t)h_crackShader;

        SplatShader h_stencilShader;
        cudaMemcpyFromSymbol(&h_stencilShader, stencilShader, shaderMemorySize);
        shaderMap["Stencil"] = (int64_t)h_stencilShader;

        SplatShader h_crackNoReconShader;
        cudaMemcpyFromSymbol(&h_crackNoReconShader, crackNoReconShader, shaderMemorySize);
        shaderMap["CrackNoRecon"] = (int64_t)h_crackNoReconShader;

        SplatShader h_roughnessOnly;
        cudaMemcpyFromSymbol(&h_roughnessOnly, roughnessOnly, shaderMemorySize);
        shaderMap["RoughnessOnly"] = (int64_t)h_roughnessOnly;

        SplatShader h_quantizeFlats;
        cudaMemcpyFromSymbol(&h_quantizeFlats, quantizeFlats, shaderMemorySize);
        shaderMap["QuantizeFlats"] = (int64_t)h_quantizeFlats;

        SplatShader h_quantizeLight;
        cudaMemcpyFromSymbol(&h_quantizeLight, quantizeLight, shaderMemorySize);
        shaderMap["QuantizeLight"] = (int64_t)h_quantizeLight;

        return shaderMap;
    }

    __global__ void ExecuteSplatShaderCUDA(SplatShader shader, int* d_splatIndexes, PackedSplatShaderParams packedParams){
        auto shaderInstance = cg::this_grid().thread_rank();
        if (shaderInstance >= packedParams.P)
            return;

        // Figure out which splat to execute on
        int idx = d_splatIndexes[shaderInstance];

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        SplatShaderConstantInputs in(packedParams, idx);
        SplatShaderModifiableInputs io(packedParams, idx);
        SplatShaderOutputs out(packedParams, idx);

        // Debug print statement for seeing what's going on inside shader kernels.
        //if (idx == 1)
            //printf("Position: (%f, %f, %f); Depth:%f; Pixel Coord: (%f, %f)\n", params.position.x, params.position.y, params.position.z, params.prerendered_depth_buffer[0], params.screen_position.x, params.screen_position.y);

        // Execute shader instance.

        shader(in, io, out);
    }
}
