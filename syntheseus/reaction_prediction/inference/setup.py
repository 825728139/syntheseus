"""Build script for simpretro_match C++ extension.

Compiles match.cpp with pybind11, linking against:
- librdchiral.so (rdchiral C++ template matching)
- RDKit libraries (molecule parsing, scoring)
"""
import os
import sysconfig
from pybind11.setup_helpers import build_ext
from setuptools import setup, Extension

# Conda environment paths
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "/home/liwenlong/miniconda3/envs/syntheseus-full-mic")

include_dirs = [
    # pybind11 headers (bundled with torch)
    f"{CONDA_PREFIX}/lib/python3.9/site-packages/torch/include",
    # RDKit headers (includes use <rdkit/GraphMol/...> and <RDGeneral/...>)
    f"{CONDA_PREFIX}/include/rdkit",
    f"{CONDA_PREFIX}/include",
    # rdchiral headers (rdchiral/rdchiral.hpp)
    f"{CONDA_PREFIX}/include",
    # Python headers
    sysconfig.get_path("include"),
]

library_dirs = [
    f"{CONDA_PREFIX}/lib",
]

libraries = [
    # rdchiral C++ library
    "rdchiral",
    # RDKit libraries (matching rdchiral_cpp/CMakeLists.txt)
    "RDKitChemReactions",
    "RDKitSmilesParse",
    "RDKitGraphMol",
    "RDKitSubstructMatch",
    "RDKitRDGeneral",
]

extra_compile_args = ["-O3", "-std=c++17", "-Wno-unused-variable", "-Wno-sign-compare"]
extra_link_args = [f"-Wl,-rpath,{CONDA_PREFIX}/lib"]

module = Extension(
    "simpretro_match",
    sources=["match.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="simpretro_match",
    ext_modules=[module],
    cmdclass={"build_ext": build_ext},
)
