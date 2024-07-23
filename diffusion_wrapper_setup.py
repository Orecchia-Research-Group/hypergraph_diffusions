import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11 until it is actually installed,
    so that the ``get_include()`` method can be invoked. """
    def __str__(self):
        import pybind11
        return pybind11.get_include()


# Set the CXX environment variable to g++
os.environ["CXX"] = "g++"

ext_modules = [
    Extension(
        'diffusion',
        ['diffusion_wrapper.cpp', 'diffusion.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-O3', '-Wall',  '-Wextra', '-std=c++20']
    ),
]

setup(
    name='diffusion',
    version='0.0.1',
    author='Konstantions Ameranis',
    author_email='kameranis@uchicago.edu',
    description='A Python wrapper for the hypergraph diffusion library',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
