#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages, Command
from sys import platform as _platform
from shutil import rmtree
import sys
import os

here = os.path.abspath(os.path.dirname(__file__))
NAME = 'PoemGenerator'

about = {}

with open(os.path.join(here, 'src', '__init__.py')) as f:
    exec(f.read(), about)

with open('README.md') as readme_file:
    readme = readme_file.read()