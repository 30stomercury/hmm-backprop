from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='hmm_forward_cpp',
    ext_modules=[
        CppExtension('hmm_forward_cpp', ['hmm_forward.cpp', 'log_matmul_kernel.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
    
