# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cknn',
    version='0.1.0',
    description='Implementation of Continuous k-Nearest Neighbors',
    long_description=readme,
    author='Naoto MINAMI',
    author_email='minami.polly@gmail.com',
    url='https://github.com/chlorochrule/cknn',
    license=license,
    packages=find_packages(exclude=('examples'))
)
