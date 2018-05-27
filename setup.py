from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

DESCRIPTION = 'Python implementations of the k-modes and k-prototypes \
               kmodesing algorithms for kmodesing categorical data.'
VERSION = '0.8-cython'

setup(
    name='kmodes',
    packages=find_packages(exclude=[
        '*.tests',
        '*.tests.*',
    ]),
    version=VERSION,
    url='https://github.com/nicodv/kmodes',
    author='Nico de Vos',
    author_email='njdevos@gmail.com',
    license='MIT',
    description=DESCRIPTION,
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        # Note: minimum numpy and scipy versions should ideally be the same
        # as what scikit-learn uses, but versions of numpy<1.10.4
        # give import problems.
        # scikit-learn version is capped to avoid compatibility issues.
        'numpy>=1.10.4',
        'scikit-learn>=0.19.0',
        'scipy>=0.13.3',
        'cython',
    ],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering'],
    ext_modules=cythonize([Extension('kmodes._k_proto',
                                     ['kmodes/_k_proto.pyx'],
                                     include_dirs=[numpy.get_include()],
                                     extra_compile_args=['-O3',
                                                         '-Wno-unreachable-code']),
                           Extension('kmodes.util._util',
                                     ['kmodes/util/_util.pyx'],
                                     include_dirs=[numpy.get_include()],
                                     extra_compile_args=['-O3',
                                                         '-Wno-unreachable-code']),
                           Extension('kmodes.util._dissim',
                                     ['kmodes/util/_dissim.pyx'],
                                     include_dirs=[numpy.get_include()],
                                     extra_compile_args=['-O3',
                                                         '-Wno-unreachable-code'])],
                          ),
)
