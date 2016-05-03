""" Setup script for pyfive. """
from setuptools import setup, find_packages

# get the long descriptions from the README.rst file
with open('README.rst') as f:
    long_description = f.read()

# get the version from the __init__.py file
with open('pyfive/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

setup(
    name='pyfive',
    version=version,
    description='A pure python HDF5 file reader',
    long_description=long_description,
    url='https://github.com/jjhelmus/pyfive',
    author='Jonathan J. Helmus',
    author_email='jjhelmus@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
    ],
    packages=['pyfive'],
    install_requires=['numpy'],
)
