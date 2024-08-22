#include "texture.h"
#include "third_party/lodepng/lodepng.h"
#include "cuda_rasterizer/auxiliary.h"
#include <iostream>
#include <stdexcept>


__global__ void PrintFirstPixel(cudaTextureObject_t texObj){
    uchar4 cudaTexel = tex2D<uchar4>(texObj, 0, 0);
    printf("Cuda reading RGBA value of first texel: %d,%d,%d,%d\n", cudaTexel.x, cudaTexel.y, cudaTexel.z, cudaTexel.w);
}


// load texture
// returns poiner to cuda textureObject? or just raw pointer to image data?
// Adapted from the lodepng decoding example example at https://github.com/lvandeve/lodepng/blob/master/examples/example_decode.cpp
// and the cuda example at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
// remember to free the 
cudaTextureObject_t* LoadTexture(std::string texturePath){
    
    //TODO: Write support for both RGB and RGBA pngs.

    std::vector<unsigned char> imageVec;
    unsigned int width, height;

    // Image is loaded as an array of interleaved chars on the format R1, G1, B1, A1, R2...
    // Each char has a range of 256 (8 bits) and represents either R,G,B or A, totalling 32 bits pr. pixel.
    // It seems that decoded RGB pictures automatically get an opaque A channel interleaved.
    unsigned int error = lodepng::decode(imageVec, width, height, texturePath);

    // Print any error that may appear during loading.
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    std::cout << "width: " << width << ", height: " << height << "\n" << std::endl;
    std::cout << "Values read in image: " << imageVec.size() << std::endl;
    std::cout << "lodepng reading RGBA value of first pixel: " << (int)imageVec[0] << "," << (int)imageVec[1] << "," << (int)imageVec[2] << "," << (int)imageVec[3] << "," << std::endl;
 
    // cast image vector to a char4 array so we can store it on cuda device as a texture
    uchar4* image_data = (uchar4*)&imageVec[0];

    printf("Host reading RGBA value of first texel as char4: %d,%d,%d,%d\n", image_data->x, image_data->y, image_data->z, image_data->w);

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(uchar4);
    // Copy data located at address image_data in host memory to device memory
    checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, image_data, spitch, width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    //TODO: Texobj should be stored in device memory instead.
    cudaTextureObject_t texObj;
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    PrintFirstPixel<<<1,1>>>(texObj);
 
    // Free device memory
    cudaFreeArray(cuArray); // We might not be able ot delete this in case it also removes all the underlying data from the textureobject.

    //This will return an invalid pointer, since texObj is just a local variable. I won't fix this, though, since there's not actually a way to keep the textrure
    //in memory after the call returns to python. This entire file is probably going to be deleted after I archive it in a commit. 
    return &texObj;
}


// unload texture

