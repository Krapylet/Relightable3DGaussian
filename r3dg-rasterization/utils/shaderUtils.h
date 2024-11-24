#pragma once
#include <cuda.h>
#include "cuda_runtime.h"

#ifndef GLM_FORCE_CUDA
    #define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

__device__ glm::vec3 RgbToHsv(glm::vec3 rgb);
__device__ glm::vec3 HsvToRgb(glm::vec3 hsv);

// Apply sobel filter to depth texture in order to generate internal outlines.
__device__ float GaussianBlur(float* sourceTexture, int pixelID, int texHeight, int texWidth);
__device__ glm::vec3 GaussianBlur(glm::vec3* sourceTexture, int pixelID, int texHeight, int texWidth);

// helper function for converting 2D pixel cooridnates to 1D pixel IDs
__device__ int GetPixelIdFromCoordinates(int x, int y, int screenWidth);

// helper function for converting 1D pixel IDs to to 2D pixel coordinates
__device__ glm::ivec2 GetPixelCoordinatesFromId(int id, int screenWidth);

// clamp pixel coordinates to be inside screen area.
__device__ glm::ivec2 ClampPixelToScreen(int x, int y, int height, int width);

// Performs a very simple quantization of the model colors
__device__ glm::vec3 Quantize(glm::vec3 input, int steps);

// Performs a very simple quantization of the model colors
__device__ float Quantize(float input, int steps);