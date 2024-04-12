#include "shader.h"
#include "config.h"
#include <cooperative_groups.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>


namespace cg = cooperative_groups;

namespace CudaShader
{

    
    // Runs after preprocess but before renderer. Allows changing values for individual splats.
    template<int C>
        __device__ static void shadeCUDA(shaderParams p)
        {
            // calculate indexes for the gaussian
            auto idx = cg::this_grid().thread_rank();
            if (idx >= p.splatsInShader)
                return;
            
            int featureIdx = idx * p.S;
            int colorIdx = idx * C;
            int posIdx = idx * 3; // mult with 3 because of x, y and z

            // TODO: make everything either glm::vec3 or float3 for consistency.
            // Glm has a lot of inbuilt functionality, but float3 is a native cuda type and probably pretty fast.
            glm::vec3 pointWorldMedian = {p.orig_points[posIdx], p.orig_points[posIdx + 1], p.orig_points[posIdx + 2]};
            glm::vec2 pointScreenMedian = {p.points_xy_image[idx].x, p.points_xy_image[idx].y};
            glm::vec3 normal = {p.features[featureIdx + 3], p.features[featureIdx + 4], p.features[featureIdx + 5]};
            glm::vec3 cameraPos = {p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]};

            // Get angle between splat and camera:
            glm::vec3 directionToCamera = cameraPos - pointWorldMedian;
            float angle = 1 - abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(normal)));
            // easing from https://easings.net/#easeInOutQuint
            float opacity = angle < 0.5
                ? 1 - 16 * pow(angle, 5)
                : pow(-2 * angle + 2, 5) / 2;

            // Set output color
            p.out_color[colorIdx + 0] = p.colors[colorIdx + 0 ] * opacity;
            p.out_color[colorIdx + 1] = p.colors[colorIdx + 1 ] * opacity;
            p.out_color[colorIdx + 2] = p.colors[colorIdx + 2 ] * opacity;
        }

    ///// Assign all the shaders to their short handles.
    shader outlineShader = &shadeCUDA<NUM_CHANNELS>;

    __global__ void ExecuteShader(shader shader, shaderParams params){
        // No need to dereference function pointers.
        
        shadeCUDA<3>(params);
    }

}

