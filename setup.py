# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    README = f.read()

with open('LICENSE') as f:
    LICENSE = f.read()

setup(
    name='cknn',
    version='0.1.0',
    description='Implementation of Continuous k-Nearest Neighbors',
    long_description=README,
    author='Naoto MINAMI',
    author_email='minami.polly@gmail.com',
    url='https://github.com/chlorochrule/cknn',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'seaborn'],
    license=LICENSE,
    packages=find_packages(exclude=('examples', '*.tests'))
)
