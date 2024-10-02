#include "texture.h"
#include "../cuda_rasterizer/auxiliary.h"
#include <iostream>
#include <stdexcept>
#include <map>
#include <cooperative_groups.h>
#include "charOperations.h"


namespace Texture
{
    // Allocates a new array from the input array where every 4th index is a padded value of 1. The input pointer is overwritten with the pointer to the new array.
    // The array is 4/3rds the length of the input array. Remember to delete the allocated array.
    __global__ void CreatPaddedArrayFromBase(float* src, float* dest, int paddedDataCount){
        auto padded_i = cooperative_groups::this_grid().thread_rank();
        if (padded_i >= paddedDataCount)
            return;

        bool padCurrentIndex = (padded_i + 1) % 4 == 0;
        if (padCurrentIndex)
        {
            dest[padded_i] = 1;
        }
        else{
            // paddedDataCount = height * width * 4
            // srcDataCount = height * width * 3
            int src_i = ceil(padded_i*0.75);
            dest[padded_i] = src[src_i];
        }
    }

    // Encodes a string representing a texture mode to a TextureMode enum
    int EncodeTextureMode(std::string mode){
        if(mode == "1")
            return TextureMode::One;
        else if ( mode == "L")
            return TextureMode::L;
        else if ( mode == "P")
            return TextureMode::P;
        else if ( mode == "RGB")
            return TextureMode::RGB;
        else if ( mode == "RGBA")
            return TextureMode::RGBA;
        else if ( mode == "CMYK")
            return TextureMode::CMYK;
        else if ( mode == "YCbCr")
            return TextureMode::YCbCr;
        else if ( mode == "LAB")
            return TextureMode::LAB;
        else if ( mode == "HSV")
            return TextureMode::HSV;
        else if ( mode == "I")
            return TextureMode::I;
        else if ( mode == "F")
            return TextureMode::F;

        return TextureMode::Unknown;
    }

    // Encodes string to cuda enum. Possible modes are:
    // - "Wrap": UVs outside the range wraps back around from the other side, repeating the texture.
    // - "Mirror": UBs outside the range wraps back from the same side, repeating the texture, but mirrored.
    // - "Clamp": UVs outside the range are clamped back into the range of the texture
    // - "Border": UVs outside the range return 0.
    int EncodeWrapMode(std::string mode){
        if(mode == "Border")
            return cudaAddressModeBorder;
        if(mode == "Clamp")
            return cudaAddressModeClamp;
        if(mode == "Mirror")
            return cudaAddressModeMirror;
        if(mode == "Wrap")
            return cudaAddressModeWrap;

        return -1;
    }

    // NOTICE: Returns a cudaTextureObject_t* cast to a int64_t
    // Creates a textureObject wrapper around the provided texture data and allocates it in memory.
    // Used in order to intialize textures outside of render loop.
    int64_t AllocateTexture(std::map<std::string, torch::Tensor> textureData){
        cudaTextureObject_t* texObj = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t));

        CreateTexture(texObj, textureData);

        return (int64_t)texObj;
    }

    // Creates a textureObject wrapper around the provided texture data
    // Adapted from the cuda example at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData){
        // extract all the texture data
        int height = textureData["height"].const_data_ptr<int>()[0];
        int width = textureData["width"].const_data_ptr<int>()[0];
        TextureMode mode = static_cast<TextureMode>(textureData["encoding_mode"].const_data_ptr<int>()[0]);
        float* pixelData = textureData["pixelData"].contiguous().cuda().mutable_data_ptr<float>();
        int pixelDataCount = textureData["pixelData"].numel();
        const cudaTextureAddressMode* addressModes = (const cudaTextureAddressMode*)textureData["wrap_modes"].const_data_ptr<int>();

        /*
        Create a channel descriptor appropriate to the image mode.
        Pillow supported channel modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, grayscale)
        P (8-bit pixels, mapped to any other mode using a color palette)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
        LAB (3x8-bit pixels, the L*a*b color space)
        HSV (3x8-bit pixels, Hue, Saturation, Value color space)
        Hue’s range of 0-255 is a scaled version of 0 degrees <= Hue < 360 degrees
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        */
        cudaChannelFormatDesc channelDesc;
        int paddedDataSize;
        int channelByteWidth;
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        if(mode == TextureMode::One || mode == TextureMode::L || mode == TextureMode::P || mode == TextureMode::I || mode == TextureMode::F){
            channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float1);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            printf("Single\n");
        }
        else if ( mode == TextureMode::RGBA || mode == TextureMode::CMYK)
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            printf("RGBA\n");
        }
        else if ( mode == TextureMode::RGB || mode == TextureMode::YCbCr)
        {
            printf("RGB\n");
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3, so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataSize);
            int paddedDataCount = width*height*4;
            CreatPaddedArrayFromBase<<<(paddedDataCount + 255) / 256, 256>>>(pixelData, paddedData, paddedDataCount);
            cudaDeviceSynchronize();
            pixelData = paddedData; // Overwrite the original data pointer. Remember to free the memory by the end of the function.
        }
        else if (mode == TextureMode::LAB|| mode == TextureMode::HSV)
        {
            printf("HSV\n");
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModePoint;  // Linear filtering is only available for floats 
            texDesc.readMode = cudaReadModeNormalizedFloat; //Notice: This might actually not work with 32 bit channels. Only 16 and 8 bit channels. But I haven't tested it.

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3m so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataSize);
            int paddedDataCount = width*height*4;
            CreatPaddedArrayFromBase<<<(paddedDataCount + 255) / 256, 256>>>(pixelData, paddedData, paddedDataCount);
            cudaDeviceSynchronize();
            pixelData = paddedData; // Overwrite the original data pointer. Remember to free the memory by the end of the function.
        }
        // Specify remaining texture object parameters
        texDesc.addressMode[0] = addressModes[0];
        texDesc.addressMode[1] = addressModes[1];
        texDesc.normalizedCoords = 1;                           //TODO: allow the coordinate mode ot be set based on a texture import setting

        // Allocate CUDA array in device memory
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);

        // Set pitch of the source (the width in memory in bytes of the 2D array pointed to by src, including padding). We dont have any padding, so it's just equal to the byte width.
        const size_t pitch = channelByteWidth;
        checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, pixelData, pitch, channelByteWidth, height, cudaMemcpyDeviceToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        
        // Create texture object, which is used as a wrapper to access the cuda Array with the actual image data.
        checkCudaErrors(cudaCreateTextureObject(texObjPtr, &resDesc, &texDesc, NULL));

        if ( mode == TextureMode::RGB || mode == TextureMode::YCbCr || mode == TextureMode::LAB || mode == TextureMode::HSV)
        {   
            // If we had to copy and pad the data of a 3-value format with a 4th value before the data was copied to a cudaArray,
            // we have to free the memory used to create the temporary padded version. 
            cudaFree(pixelData);
        }
    }

    // NOTICE: Returns a std::map<std::string, std::map<std::string, cudaTextureObject_t*>>* cast to an int64_t in order to get around the pybind pointer wierdness.
    // Takes the texture tensor Maps and uses it to create wrapper objects around the texture data, so it can be accessed efficiently in the shaders.
    // shaderTextureTensorMaps stores data in nested maps on the format: <ShaderName, <TextureName, <TexturePropertyName, TexturePropertyData*>>>
	// shaderTextureMaps stores data in nested maps on the format: <ShaderName, <TextureName, TextureObject*>>
    int64_t InitializeTextureMaps(
        const std::map<std::string, std::map<std::string, std::map<std::string, torch::Tensor>>>& shaderTextureTensorMaps)
    {
        auto shaderTextureMaps = new std::map<std::string, std::map<std::string, cudaTextureObject_t*>>;

        for(auto shaderTextureTensorBundle : shaderTextureTensorMaps){
            std::string shaderName = shaderTextureTensorBundle.first;
            auto textureTensorBundle = shaderTextureTensorBundle.second;

            for(auto textureTensor : textureTensorBundle){
                std::string textureName = textureTensor.first;
                auto textureData = textureTensor.second;

                cudaTextureObject_t* texObj = (cudaTextureObject_t*) malloc(sizeof(cudaTextureObject_t)); // TODO: Does this actually need to be malloced? 
                Texture::CreateTexture(texObj, textureData);
                (*shaderTextureMaps)[shaderName][textureName] = texObj;
            }
        }

        //std::cout << "ShaderTextureBundle poiter" << shaderTextureMaps << ". Casting to " << (int64_t)shaderTextureMaps << std::endl;
        return (int64_t)shaderTextureMaps;
    }

    // Frees the underlying cudaArray that the textureObject is wrapped around, as well as the texture object pointer that contains it.
    void UnloadTexture(cudaTextureObject_t* textureObject){
        cudaResourceDesc resDesc;
        checkCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, (*textureObject)));
        checkCudaErrors(cudaFreeArray(resDesc.res.array.array));
    }

    // Frees the underlying cudaArray that the textureObject is wrapped around
    void UnloadTexture(cudaTextureObject_t textureObject){
        cudaResourceDesc resDesc;
        checkCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, textureObject));
        checkCudaErrors(cudaFreeArray(resDesc.res.array.array));
    }

    // NOTICE: takes a std::map<std::string, std::map<std::string, cudaTextureObject_t*>>* that has been cast to an int64_t in order to get around the pybind pointer wierdness.
    // Unloads all the memory allocated for all the texture Maps, including the input pointer.
    void UnloadTextureMaps (int64_t shaderTextureMaps_mapPtr){
        auto shaderTextureMaps = (std::map<std::string, std::map<std::string, cudaTextureObject_t*>>*)shaderTextureMaps_mapPtr;

        // For each shader, unload all textures from memory
        for(auto shaderTextureBundle : (*shaderTextureMaps)){
            auto textureBundle = shaderTextureBundle.second;
            
            for(auto texture : textureBundle){
                cudaTextureObject_t* texObj = texture.second;
                Texture::UnloadTexture(texObj);
            }
        }

        delete(shaderTextureMaps);
    }

    // initialize device texture vector and device texture name vector (used for indirect addressing of textures)
    // NOTICE: actually returns a d_TextureManager* cast to an int64.
    int64_t UploadTexturesToDevice(std::vector<std::string> names, std::vector<int64_t> textureObjects){
        auto h_texManager = new TextureManager();
        h_texManager->SetTextures(names, textureObjects);
        // TODO: Set error textures.

        TextureManager* d_texManager;
        cudaMalloc(&d_texManager, sizeof(TextureManager));
        cudaMemcpy(d_texManager, h_texManager, sizeof(TextureManager), cudaMemcpyHostToDevice);

        return (int64_t)d_texManager;
    }

    // --------------- Debug methods ---------------

    // Test whether we can do the texture initialization before the call.
    // Returns an int* cast to an int64_t in order to get around pybind wierdness.
    int64_t AllocateVariable(){
        int* allocedPointer = (int*) malloc(sizeof(int)); 
        (*allocedPointer) = 10;
        std::cout << "C++ pointer saved at " << allocedPointer << std::endl;
        return (int64_t)allocedPointer;
    }

    void PrintVariable (int64_t allocedPointer_intPtr){
        std::cout << "Reading following value from alloced pointer: " << (*((int*)allocedPointer_intPtr)) << std::endl;
    }

    void DeleteVariable(int64_t allocatedPointer_intPtr){
        std::cout << "Deleting allocated pointer" << std::endl;
        delete (int*)allocatedPointer_intPtr;
        std::cout << "Deleting done" << std::endl;
    }

    // Debug Method used for quickly testing whether 
    __global__ void PrintFirstPixel(cudaTextureObject_t texObj){
        float4 cudaTexel = tex2D<float4>(texObj, 0, 0);
        printf("Cuda reading RGBA value of first texel: %f,%f,%f,%f\n", cudaTexel.x, cudaTexel.y, cudaTexel.z, cudaTexel.w);
    }

    __global__ void PrintFromTextureManagerCUDA(TextureManager *texManager, char* targetName){
        cudaTextureObject_t texObj = texManager->GetTexture(targetName);
        float4 cudaTexel = tex2D<float4>(texObj, 0, 0);
        printf("Cuda reading RGBA value of first texel in tex %lu-'%s': %f,%f,%f,%f\n", texObj, targetName, cudaTexel.x, cudaTexel.y, cudaTexel.z, cudaTexel.w);
    }

    void PrintFromTextureManager(int64_t texManager_ptr, std::string targetName){
        auto d_texManager = (TextureManager*)texManager_ptr;

        // transfer name to device
        char* d_targetName;
        cudaMalloc(&d_targetName, (targetName.length()+1)*sizeof(char)); // Add 1 char to have room for termination character added by c_str()
        cudaMemcpy(d_targetName, targetName.c_str(), targetName.length()+1, cudaMemcpyKind::cudaMemcpyHostToDevice);

        PrintFromTextureManagerCUDA<<<1,1>>>(d_texManager, d_targetName);

        // Delete the name that was transfered to device.
        cudaFree(d_targetName);
        cudaDeviceSynchronize();
    }
}

/// -----------------------Texture manager class implementation -------------------

    __host__ Texture::TextureManager::TextureManager(){};

    // Allocates and Uploads an array an array of textures onto the GPU so that textures can be looked up by in the shaders.
    __host__ void Texture::TextureManager::SetTextures(std::vector<std::string> names, std::vector<int64_t> textureObjects){
        // First, move each element into a vector on host
        // We use vectors instead of arrays since we don't know the size at compile time, and we don't wanna allocate memory ourselves.
        int h_texCount = names.size();
        std::vector<char*> h_names (h_texCount);
        std::vector<cudaTextureObject_t> h_texObjs(h_texCount);

        for (size_t i = 0; i < h_texCount; i++)
        {
            std::string name = names[i];
            cudaTextureObject_t texObj = *((cudaTextureObject_t*)textureObjects[i]);

            // convert each name to a char array located in device memory
            // Texture objects are already in device memory, so we don't need to do anything to them.
            int stringlength = name.length();
            char* d_charName;
            cudaMalloc(&d_charName, (stringlength+1)*sizeof(char)); // add 1 to also include the termination character
            cudaMemcpy(d_charName, name.c_str(), stringlength+1, cudaMemcpyKind::cudaMemcpyDefault);

            // Save the pointers to the tempoary host vectors.
            h_names[i] = d_charName;
            h_texObjs[i] = texObj;
        }
        // Then allocate all the arrays on the device and transfer the data stored on host to them (and to the texCount variable):
        printf("Trying to store %i as tex count\n", h_texCount);
        cudaMalloc(&d_texCount, sizeof(int));
        cudaMemcpy(d_texCount, &h_texCount, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_textureNames, h_texCount * sizeof(char*));
        cudaMemcpy(d_textureNames, &h_names[0], h_texCount *sizeof(char*), cudaMemcpyHostToDevice);

        cudaMalloc(&d_textureObjects, h_texObjs.size() * sizeof(cudaTextureObject_t));
        cudaMemcpy(d_textureObjects, &h_texObjs[0], h_texObjs.size() * sizeof(cudaTextureObject_t), cudaMemcpyKind::cudaMemcpyDefault);
    }
            
    // Deallocates all textures on the device (except the error texture)
    __host__ void Texture::TextureManager::UnloadTextures(){
        for (size_t i = 0; i < *d_texCount; i++)
        {
            char* name = d_textureNames[i];
            cudaTextureObject_t texObj = d_textureObjects[i];

            delete(name);
            UnloadTexture(texObj);
        }
    }

    // Allocates and uploads an errot texture to the device.
    __host__ void Texture::TextureManager::SetErrorTexture(cudaTextureObject_t errorTexture){
        cudaMalloc(&d_textureObjects, sizeof(cudaTextureObject_t));
        cudaMemcpy(&d_errorTexture, &errorTexture, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }

    //Deallocates the error texture on the device.
    __host__ void Texture::TextureManager::UnloadErrorTexture(){
        UnloadTexture(d_errorTexture);
    }

    // Returns the error texture.
    __device__ cudaTextureObject_t Texture::TextureManager::GetErrorTexture(){return *d_errorTexture;}
            
    // Loops through each loaded texture name and checks if it matches the given name before returning the associated texture. Is pretty slow, so cache the result.
    // Returns the error texture if no texture is found.
    __device__ cudaTextureObject_t Texture::TextureManager::GetTexture(char* targetTextureName){
        for (size_t i = 0; i < *d_texCount; i++)
        {
            // Check if the name in the lookup table is the same as the target name
            char* currentTexName = d_textureNames[i];
            bool textureHasBeenFound = charsAreEqual(currentTexName, targetTextureName);
            
            if(textureHasBeenFound){
                cudaTextureObject_t texObj = d_textureObjects[i];
                return texObj;
            }
        }

        // If no texture was found with the given name, return the default error texture
        printf("Warning: Could not find texture '%s'\n", targetTextureName);
        return *d_errorTexture;
    }

    