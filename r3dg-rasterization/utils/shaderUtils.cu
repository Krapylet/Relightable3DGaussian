#include "shaderUtils.h"


// Converts an RGB vector in the range [0-1] to a HSV vector in the range [0-1]
// Based off of the nvidia CUDA sample at https://github.com/hellopatrick/cuda-samples/blob/master/hsv/kernel.cu
__device__ glm::vec3 RgbToHsv(glm::vec3 rgb) {	
    float max = fmax(rgb.r, fmax(rgb.g, rgb.b));
	float min = fmin(rgb.r, fmin(rgb.g, rgb.b));
	float diff = max - min;
    
	glm::vec3 hsv;
    float* h = &hsv[0];
    float* s = &hsv[1];
    float* v = &hsv[2];
	*v = max;
	
	if(*v == 0.0f) { // black
		*h = *s = 0.0f;
	} else {
		*s = diff / *v;
		if(diff < 0.001f) { // grey
			*h = 0.0f;
		} else { // color
			if(max == rgb.r) {
				*h = (rgb.g - rgb.b)/diff/6;
				if(*h < 0.0f) { *h += 1.0f; }
			} else if(max == rgb.g) {
				*h = (2 + (rgb.b - rgb.r)/diff)/6;
			} else {
				*h = (4 + (rgb.r - rgb.g)/diff)/6;
			}
		}		
	}
	
	return hsv;
}

// Converts a HSV vector in the range [0-1] to an RGB vector in the range [0-1]
// Based off of the nvidia CUDA sample at https://github.com/hellopatrick/cuda-samples/blob/master/hsv/kernel.cu
__device__ glm::vec3 HsvToRgb(glm::vec3 hsv) {
    float h = hsv[0];
    float s = hsv[1];
    float v = hsv[2];

    glm::vec3 rgb;
    float* r = &rgb[0];
    float* g = &rgb[1];
    float* b = &rgb[2];
		
	float f = h*6;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	
	if(hi == 0.0f || hi == 6.0f) {
		*r = v;
		*g = t;
		*b = p;
	} else if(hi == 1.0f) {
		*r = q;
		*g = v;
		*b = p;
	} else if(hi == 2.0f) {
		*r = p;
		*g = v;
		*b = t;
	} else if(hi == 3.0f) {
		*r = p;
		*g = q;
		*b = v;
	} else if(hi == 4.0f) {
		*r = t;
		*g = p;
		*b = v;
	} else {
		*r = v;
		*g = p;
		*b = q;
	}
	
	return rgb;
}

// Matrix that can be used for gaussian blurs. Isn't actually a real gaussian matrix, as i just apprixmated it manually.
__device__ float BlendingMatrix[5][5] = 
{{0.009375f, 0.01875f, 0.028125f, 0.01875f, 0.009375f},
{0.01875f, 0.0375f, 0.045f, 0.0375f, 0.01875f},
{0.028125f, 0.045f, 0.3f, 0.045f, 0.028125f},
{0.01875f, 0.0375f, 0.045f, 0.0375f, 0.01875f},
{0.009375f, 0.01875f, 0.028125f, 0.01875f, 0.009375f}};

// Apply sobel filter to depth texture in order to generate internal outlines.
__device__ float GaussianBlur(float* sourceTexture, int pixelID, int texHeight, int texWidth){
	
	float blurredPixel = 0;
	for (int x = -2; x < 3; x++)
	{
		for (int y = -2; y < 3; y++)
		{
			int sample_pid = pixelID + x + y * texWidth;
			sample_pid = max(0, min(texHeight * texWidth-1, sample_pid)); // Clamp sample to inside texture. Doesn't wraps around texture instead of clamping to corners.
			float texSample = sourceTexture[sample_pid];
			blurredPixel += BlendingMatrix[x+2][y+2] * texSample;
		}
	}

	return blurredPixel;
}

__device__ glm::vec3 GaussianBlur(glm::vec3* sourceTexture, int pixelID, int texHeight, int texWidth){
	
	glm::vec3 blurredPixel(0);
	for (int x = -2; x < 3; x++)
	{
		for (int y = -2; y < 3; y++)
		{
			int sample_pid = pixelID + x + y * texWidth;
			sample_pid = max(0, min(texHeight * texWidth-1, sample_pid)); // Clamp sample to inside texture. Doesn't wraps around texture instead of clamping to corners.
			glm::vec3 texSample = sourceTexture[sample_pid];
			blurredPixel += BlendingMatrix[x+2][y+2] * texSample;
		}
	}

	return blurredPixel;
}

// helper function for converting 2D pixel cooridnates to 1D pixel IDs
__device__ int GetPixelIdFromCoordinates(int x, int y, int screenWidth){
	return x + y * screenWidth;
}

// helper function for converting 1D pixel IDs to to 2D pixel coordinates
__device__ glm::ivec2 GetPixelCoordinatesFromId(int id, int screenWidth){
	return glm::ivec2(id % screenWidth, id / screenWidth);
}

__device__ glm::ivec2 ClampPixelToScreen(int x, int y, int height, int width){
	int clamped_x = max(0, min(x, width));
	int clamped_y = max(0, min(y, height));
	return glm::vec2(clamped_x, clamped_y);
}

// Performs a very simple quantization of the model colors
__device__ glm::vec3 Quantize(glm::vec3 input, int steps)
{
	// for each component of the color, clamp it to the closest multiple of the step threshold (1/steps).
	float quantizedR = roundf(input.r * steps)/steps;
	float quantizedG = roundf(input.g * steps)/steps;
	float quantizedB = roundf(input.b * steps)/steps;

	glm::vec3 quatnizedColor(quantizedR, quantizedG, quantizedB);
	return quatnizedColor;
}

// Performs a very simple quantization of the model colors
__device__ float Quantize(float input, int steps)
{
	// clamp it to the closest multiple of the step threshold (1/steps).
	return roundf(input * steps)/steps;
}