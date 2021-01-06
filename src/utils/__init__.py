from __future__ import absolute_import

from .textlibrary import build, TextLibrary
from .gutenberg import *

def build_dataset(args: dict) -> TextLibrary:
    return build(args)