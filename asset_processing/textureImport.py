from PIL import Image
import torch
import torchvision.transforms.functional as TF
import os
from r3dg_rasterization import _C

class TextureImporter:
    def __init__(self):
        self.loadedTextureNames : list[str] = []
        self.loadedTextureObjects : list[int] = []

    # Loads a texture into CUDA memory and returns a tensor pointing to the image.
    def import_texture(self, textureName:str, filepath:str, wrapModeHorizontal = "Wrap", wrapModeVertical = "Wrap", normalizedCoords = True) -> int:
        # Open image
        filename = os.path.basename(filepath)
        image = Image.open(filepath)
        
        # Create tensor from image with type float and range [0,1]. LAB and HSV images aren't rescaled
        imageTensor = TF.to_tensor(image).permute(1, 2, 0)

        # Store image data in a dictionary along with relevant metadata
        # We have to store all the other data in tensors so they can be passed to c++ through the pybind interface.
        # We don't store the metadata on device, since the c++ host needs it to construct the textureObjects.
        width, height = image.size
        encodingMode = _C.EncodeTextureMode(image.mode) #We have to encode the string as an int. Otherwise we cant read it from the tensor in c++.
        wrapModeH = _C.EncodeWrapMode(wrapModeHorizontal)
        wrapModeV = _C.EncodeWrapMode(wrapModeVertical)
        image = {
            "pixelData" : imageTensor.cuda(), #Stored on device
            #Other metedata is stored on CPU
            "height" : torch.tensor([height], dtype=torch.int32),
            "width" : torch.tensor([width], dtype=torch.int32),
            "encoding_mode" : torch.tensor([encodingMode], dtype=torch.int32), # Which format the image data is stored in: CMYK, RGBA, RGB, HSV etc.
            "wrap_modes" : torch.tensor([wrapModeH, wrapModeV], dtype=torch.int32),
            "normalizedCoords" : torch.tensor([int(normalizedCoords)], dtype=torch.int32),
        }

        # Create texObj from image tensor:
        texObj = _C.AllocateTexture(image)

        ## Add tex object to tex vector
        self.loadedTextureNames.append(textureName)
        self.loadedTextureObjects.append(texObj)

        return texObj

    def initialize_all_textures(self):
        ## Import textures
        #TODO: make the paths relative to this directory
        self.import_texture("Cracks", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Cracks 2.png")
        self.import_texture("Grainy", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Grainy 13.png")
        self.import_texture("Grid", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Grid.png")
        self.import_texture("Turbulence", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Turbulence 2.png")
        self.import_texture("Craters", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Craters 12.png")
        self.import_texture("Depth cracks", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Depth Cracks.png")
        self.import_texture("Bulge", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\bulge.png")
        self.import_texture("shadow", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\shadow.png")

        #Set error texture
        errorTex = self.import_texture("Error", r"C:\Users\asvj\Documents\GitHub\Relightable3DGaussian\textures\Error.png")

        #Once all textures have been loaded, create an indirect address lookup table for them on the device:
        d_textureManager_ptr = _C.UploadTexturesToDevice(self.loadedTextureNames, self.loadedTextureObjects, errorTex)

        return d_textureManager_ptr