from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='lattice',
    ext_modules=[CppExtension('lattice', ['Image.cpp','lattice.cpp'])],
    cmdclass={'build_ext':BuildExtension})

