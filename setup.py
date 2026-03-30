#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

REQUIREMENTS = [
    'iminuit', #fit in 'utils/functions.py'
    'matplotlib',
    'mat73', #read v7.3 mat files
    'numba', #for multiprocessing
    'numpy',
    'palettable', # nice color maps
    'pandas', #dataframe
    'psutil', #memory check
    #'pylandau',  #called in 'utils/functions.py' but works only for python <= v3.7
    'pyjson', #json files
    'pyvista',
    'pyyaml', #yaml files
    'scikit-learn', #ml library
    'scikit-image', #ransac
    'scipy', 
    'torch', #machine learning
    'tqdm', #process monitoring 
    'seaborn',#nice plot templates
    'uproot', #to read, update, and write root file
    'vtk', #3d modelling
]

setup(
    name='pymusouf',
    version='0.2.0',
    description="This package is for processing and analyzing muography data " \
                "recorded at Soufrière de Guadeloupe/Copahue volcanoes " \
                "with scintillator-based telescopes developed at IP2I, Lyon",
    author="Raphaël Bajou",
    author_email='r.bajou2@gmail.com',
    url='https://github.com/rbajou/pymusouf.git',
    packages=find_packages(), 
    package_dir={
        'pymusouf': 'pymusouf'
    },  
    include_package_data=True,
    package_data={
          'config': ["*.yaml"],
          'survey': ["*.yaml"],
     },
    install_requires=REQUIREMENTS,
    keywords=['Muography', 'Volcano', 'Scintillator'],
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ],
)


