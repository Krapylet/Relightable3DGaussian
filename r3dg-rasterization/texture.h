#pragma once

#include <string>

__global__ extern void PrintFirstPixel(cudaTextureObject_t texObj);
cudaTextureObject_t* LoadTexture(std::string texturePath);