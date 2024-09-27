#include <stdio.h>

__device__ bool charsAreEqual(const char* strA, const char* strB){
    int maxLength = 256;
    for (size_t i = 0; i < maxLength; i++)
    {
        bool hasReachedEndOfAString = (strA[i] == 0) || (strB[i] == 0);
        if (hasReachedEndOfAString){
            // If we reach the end of a string without finding a mismacht, the strings are equal of they both have equal length.
            bool stringsHaveEqualLength = strA[i] == 0 && strB[i] == 0;
            return stringsHaveEqualLength;
        }
        
        // if strings have different chars in the same positions they're not equal.
        bool FoundDifferentCharsInStrings = strA[i] != strB[i];
        if (FoundDifferentCharsInStrings){
            return false;
        }
    }

    // If no mismatch has been found after the max length, just assume they're equal.
    return true;
}