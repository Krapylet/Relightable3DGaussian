from r3dg_rasterization import _C


class PostProcessManager:
    def __init__(self):
        self.postProcessingShaders : dict[str, int] = _C.GetPostProcessShaderAddressMap()
        self.activePostProcessingPasses : list[int] = []
        
    def AddPostProcessingPass(self, passName : str):
        print(self.postProcessingShaders)

        shaderAddress = self.postProcessingShaders[passName]
        
        shaderDidNotExist : bool = shaderAddress == None
        if(shaderDidNotExist):
            print(f"Could not find post prossing pass by the name of '{passName}'. No pass was added")
            return
        
        self.activePostProcessingPasses.append(shaderAddress)
