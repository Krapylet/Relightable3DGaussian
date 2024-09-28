#include "indirectMap.h"
#include "charOperations.h"

template <typename KeyType, typename ValueType>
__host__ IndirectMap<KeyType, ValueType>::IndirectMap(){
    // Pass for now
};

template <typename ValueType>
__host__ IndirectMap<char*, ValueType>::IndirectMap(){
    // Pass for now
};

template <typename KeyType, typename ValueType>
__host__ IndirectMap<KeyType, ValueType>::IndirectMap(std::vector<KeyType> keys, std::vector<ValueType> values){
    SetAll(keys, values);
};

template <typename ValueType>
__host__ IndirectMap<char*, ValueType>::IndirectMap(std::vector<char*> keys, std::vector<ValueType> values){
    SetAll(keys, values);
};

// Allocates and Uploads keys and values to internal device arrays.
template <typename KeyType, typename ValueType>
__host__ void IndirectMap<KeyType, ValueType>::SetAll(std::vector<KeyType> keys, std::vector<ValueType> values){
    int h_keyCount = keys.size();
    cudaMalloc(&d_itemCount, sizeof(int));
    cudaMemcpy(d_itemCount, &h_keyCount, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_keys, h_keyCount * sizeof(KeyType));
    cudaMemcpy(d_keys, &keys[0], h_keyCount *sizeof(KeyType), cudaMemcpyHostToDevice);

    cudaMalloc(&d_values, h_keyCount * sizeof(ValueType));
    cudaMemcpy(d_values, &values[0], h_keyCount * sizeof(ValueType), cudaMemcpyKind::cudaMemcpyDefault);
}

// Allocates and Uploads keys and values to internal device arrays.
template <typename ValueType>
__host__ void IndirectMap<char*, ValueType>::SetAll(std::vector<char*> keys, std::vector<ValueType> values){
        int h_keyCount = keys.size();
        std::vector<char*> h_keys (h_keyCount);

        // First we have to transfer each char array individually to the GPU.
        for (size_t i = 0; i < h_keyCount; i++)
        {
            std::string name = names[i];

            int stringlength = name.length();
            char* d_charName;
            cudaMalloc(&d_charName, (stringlength+1)*sizeof(char)); // add 1 to also include the termination character
            cudaMemcpy(d_charName, name.c_str(), stringlength+1, cudaMemcpyKind::cudaMemcpyDefault);

            // Save the pointers to the tempoary host vectors.
            h_keys[i] = d_charName;
        }

        // Then allocate all the arrays on the device and transfer the data stored on host to them (and to the texCount variable):
        cudaMalloc(&this->d_itemCount, sizeof(int));
        cudaMemcpy(this->d_itemCount, &h_keyCount, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&this->d_keys, h_keyCount * sizeof(char*));
        cudaMemcpy(this->d_keys, &h_keys[0], h_keyCount *sizeof(char*), cudaMemcpyHostToDevice);

        cudaMalloc(&this->d_values, h_keyCount * sizeof(ValueType));
        cudaMemcpy(this->d_values, &values[0], h_keyCount * sizeof(ValueType), cudaMemcpyKind::cudaMemcpyDefault);
}

//Read from map on device
template <typename KeyType, typename ValueType>
__device__ ValueType IndirectMap<KeyType, ValueType>::operator[](KeyType key){
    for (size_t i = 0; i < *this->d_itemCount; i++)
    {
        // Check if the name in the lookup table is the same as the target name
        KeyType currentKey = this->d_keys[i];
        bool keyHasBeenFound = key == currentKey;
        
        if(keyHasBeenFound){
            ValueType value = this->d_values[i];
            return value;
        }
    }

    // If the value can't be found, we would usually throw an error, but we can't throw errors in cuda.
    // So instead we assert a false statement to make the program halt.
    assert(false);
}

//Read from map on device
template <typename KeyType, typename ValueType>
__device__ ValueType IndirectMap<KeyType, ValueType>::Get(KeyType key){
    return this[key];
}

//Read from map on device. As a special case, if the key type is a char*, we will try to compare the contents of the char array
//instead of the actual pointers.
template <typename ValueType>
__device__ ValueType IndirectMap<char*, ValueType>::operator[](char* string){
    for (size_t i = 0; i < *this->d_itemCount; i++)
    {
        // Check if the name in the lookup table is the same as the target name
        char* currentKey = this->d_keys[i];
        bool keyHasBeenFound = charsAreEqual(key, currentKey);
        
        if(textureHasBeenFound){
            ValueType value = this->d_values[i];
            return value;
        }
    }

    // If the value can't be found, we would usually throw an error, but we can't throw errors in cuda.
    // So instead we assert a false statement to make the program halt.
    assert(false);
}

template <typename ValueType>
__device__ ValueType IndirectMap<char*, ValueType>::Get(char* string){
    return this[string];
}

template <typename KeyType, typename ValueType>
__device__ IndirectMap<KeyType, ValueType>::~IndirectMap(){
    cudaFree(this->d_itemCount);
    cudaFree(this->d_keys);
    cudaFree(this->d_values);
}

template <typename ValueType>
__device__ IndirectMap<char*, ValueType>::~IndirectMap(){
    cudaFree(this->d_itemCount);
    cudaFree(this->d_keys);
    cudaFree(this->d_values);
}