#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="r3dg_rasterization",
    packages=['r3dg_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="r3dg_rasterization._C",
            sources=[
                "cuda_rasterizer/shShader.cu",
                "cuda_rasterizer/splatShader.cu",
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "render_equation.cu",
                "ext.cu",
                "preprocessModel.cu",
                "utils/texture.cu",
                "utils/charOperations.cu",
                "utils/includeTorch.cu"],
            dlink=True,
            extra_compile_args={
                "nvcc": ["--device-c",
                         "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                         "-O3", "-arch=compute_61"],
                "cxx": ["-O3"]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
