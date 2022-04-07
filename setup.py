#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   setup.py
@Time    :   2022/04/04 10:24:59
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''
from setuptools import setup, find_packages

# List of requirements
requirements = []  # This could be retrieved from requirements.txt

# Package (minimal) configuration
setup(
    name="matgnn",
    version="1.0.0",
    description="util package for matgg_perovskites",
    packages=find_packages(),  # __init__.py folders search
    install_requires=requirements
)