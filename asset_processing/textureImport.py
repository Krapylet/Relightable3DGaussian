from PIL import Image
import torch
import torchvision.transforms.functional as TF
import os
from r3dg_rasterization import _C

# Loads a texture into CUDA memory and returns a tensor pointing to the image.
def import_texture(filepath:str) -> dict[str, torch.Tensor]:
    # Open image
    filename = os.path.basename(filepath)
    image = Image.open(filepath)
    
    # Create tensor from image with type float and range [0,1]. LAB and HSV images aren't rescaled
    imageTensor = TF.to_tensor(image).cuda()

    # Store image data in a dictionary along with relevant metadata
    # We have to store all the other data in tensors so they can be passed to c++ through the pybind interface.
    # We don't store the metadata on device, since the c++ host needs it to construct the textureObjects.
    width, height = image.size
    mode = _C.encodeTextureMode(image.mode) #We have to encode the string as an int. Otherwise we cant read it from the tensor in c++.
    image = {
        "pixelData" : imageTensor, #Stored on device
        #Other metedata is stored on CPU
        "height" : torch.tensor([height], dtype=torch.int32),
        "width" : torch.tensor([width], dtype=torch.int32),
        "mode" : torch.tensor([mode], dtype=torch.int32) # Which format the image data is stored in: CMYK, RGBA, RGB, HSV etc.
    }
    return image

def initialize_all_textures():# -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    ## Import textures
    # First import textures for each of the shaders
    #TODO: make the paths relative to this directory
    ShDefaultTextures = {
        
        "Cracks": import_texture(r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Cracks 2.png"),
        "Red": import_texture(r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\redTest.png")
    }

    ExpPosTextures = {
    }

    SplatDefault = {
    }

    OutlineShaders = {
    }

    WireframeShader = {
    }

    #Then collect all the shaders into a single map (key by shader address instead?)
    ShaderTextures = {
        "ShDefault": ShDefaultTextures,
        "ExpPos": ExpPosTextures,
        "SplatDefault": SplatDefault,
        "OutlineShader": OutlineShaders,
        "WireframeShader": WireframeShader
    }
    return ShaderTextures