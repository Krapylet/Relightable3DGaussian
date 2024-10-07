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
				*h = 60.0f * (rgb.g - rgb.b)/diff;
				if(*h < 0.0f) { *h += 360.0f; }
			} else if(max == rgb.g) {
				*h = 60.0f * (2 + (rgb.b - rgb.r)/diff);
			} else {
				*h = 60.0f * (4 + (rgb.r - rgb.g)/diff);
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
		
	float f = h/60.0f;
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