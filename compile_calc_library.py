from setuptools import setup
from Cython.Build import cythonize

# Build command: python compile_calc_library.py build_ext --inplace
setup(ext_modules = cythonize("sandpile_calculations.pyx", compiler_directives={'language_level': 3}))