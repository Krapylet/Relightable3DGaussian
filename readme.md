# Shader-GS: Bringing Artistic Control to Gaussian Splatting.


<p align="center">
A Master thesis by <br>
Aske Valdemar Szlavik Johansen, asvj@itu.dk
</p>


### <p align="center"> [📰Thesis](https://1drv.ms/b/s!AnEI-QEq46P2h8Rr23pRvRhnYTH8Aw?e=fHqhX5) | [🖼️Appendix](https://1drv.ms/b/s!AnEI-QEq46P2h8Rq-xWg-HpLO6oTzg?e=ciubjD) | ~~🗿Models~~ (too big for me to host myself 😢)</p>


This is an implementation of my master thesis *Shader-GS: Bringing Artistic Control to Gaussian Splatting.* <br>
![Alt text](media/teaser.gif)

### Changes from base project
This project is built directly on top of the great [R3DG reference implementation](https://github.com/NJU-3DV/Relightable3DGaussian) by Gao et al. While I have touched more or less the entire repo at least a bit, the following files and folders make up the bulk the work i have personally done on this project:
```
# Original = My own work
# Modified = Files i have made medium to major modifiactions to.
- asset_processing/*..............................# Original
- gaussian_renderer/neilf.py......................# Modified
- gaussian_renderer/r3dg_rasterization.py.........# Modified
- r3dg-rasterization/cuda_rasterizer/*............# All shader files are original. The rest are heavily modified
- r3dg-rasterization/utils/*......................# original
- r3dg-rasterization/preprocessModel..............# original
- r3dg-rasterization/shaderManager................# original
- r3dg-rasterization/rasterize_points.............# Modified
- scene/gaussian_model.py.........................# Modified
- gui.py..........................................# Modified
```


### Install guide for windows 11
```shell
# 1. Install miniconda
#	- Add conda to path during install.

# 2. Install Visual C++ Redistributable for Visual Studio 2019
# Select the 'Desktop developmepment with c++' package to get the x64/x86 build tools

# 3. Install CUDA Toolkit 11.8
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# 4. Clone project https://github.com/Krapylet/Relightable3DGaussian

# Note Step 5 and 21 can be skipped if you use the pre-trained models from the model appendix.
# 5. download nerf dataset and place it in relightable3DGaussian/datasets/ (Takes a very long time, but you can continue the install while the data downloads)

# 6. Add following paths to PATH environment variable:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp

# 7. Add new environment virable:
# CUDA_HOME which contians the path to the cuda install. Default path is: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

# 8. open "x86_x64 cross Tools command prompt" as administrator and navigate to the repo directory

# 9. Set environment variable in terminal
SET DISTUTILS_USE_SDK=1

# 10. install environment
conda env create --file environment.yml
conda activate r3dgs

# 11. install pytorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# 12. install torch_scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu118.html
# alternatively conda install pytorch-scatter -c pyg

# 13. install kornia
pip install kornia==0.7.3

# 14. install nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

# 15. install knn-cuda
pip install ./submodules/simple-knn

# 16. install bvh
pip install ./bvh

# 17. install ninja
pip install ninja==1.11.1.1

# 18. install pyexr
pip install pyexr==0.4.0

# 19. Manually patch bug torch's cpp extension
# Guide is placed at root of this repo as "Guide to manually patching torch cpp_extension.txt"

# 20. install relightable 3D Gaussian
pip install ./r3dg-rasterization

# Note Step 5 and 21 can be skipped if you use the pre-trained models from the model appendix.
# 21. Wait for dataset download in step 5 to finish. Then train datasets to generate point clouds (Takes a very long time on RTX 2080.  Can be stopped after a single model has been trained AND evaluated)
.\script\run_nerf.sh

# 22. Initialize the c++ part of the razterization:
cd r3dg-rasterization
cd build				
cmake ../                   
cd ..
cmake --build build

# 23. Compile c++ razterization:
pip install ./r3dg-rasterization

# 24. Finally, run the r3dg on a trained model:
python gui.py -m output/NeRF_Syn/{MODELNAME}/neilf -t neilf --debug
```
