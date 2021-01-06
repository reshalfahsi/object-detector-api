from __future__ import absolute_import

from .models import PoemGenerator
from .datasets import TextLibraryDataset
from .dataset_adjustment import adjust_dataset
from .utils import TextLibrary
from .utils import GutenbergLib, create_libdesc

__all__ = [PoemGenerator, TextLibraryDataset, adjust_dataset]
__version__ = '1.0.1'