"""
TextLibrary class. 

Text library for training, encoding, batch generation, and formatted source display. 
It read some books from Project Gutenberg and supports creation of training batches. 

Mostly copy-paste from https://github.com/domschl/torch-poet
"""

import os
import random
import numpy as np
from urllib.request import urlopen

from .gutenberg import *

class TextLibrary(object):
    def __init__(self, descriptors, text_data_cache_directory=None, max=100000000):
        self.descriptors = descriptors
        self.data = ''
        self.cache_dir = text_data_cache_directory
        self.files = []
        self.c2i = {} # char to integer
        self.i2c = {} # integer to char
        index = 1
        dat = None
        for descriptor, author, title in descriptors:
            fd = {}
            cache_name = get_cache_name(self.cache_dir, author, title)
            if os.path.exists(cache_name):
                is_cached = True
            else:
                is_cached = False
            valid = False
            if descriptor[:4] == 'http' and is_cached is False:
                try:
                    print(f"Downloading {cache_name}")
                    dat = urlopen(descriptor).read().decode('utf-8')
                    if dat[0] == '\ufeff':  # Ignore BOM
                        dat = dat[1:]
                    dat = dat.replace('\r', '')  # get rid of pesky LFs
                    self.data += dat
                    fd["title"] = title
                    fd["author"] = author
                    fd["data"] = dat
                    fd["index"] = index
                    index += 1
                    valid = True
                    self.files.append(fd)
                except Exception as e:
                    print(f"Can't download {descriptor}: {e}")
            else:
                fd["title"] = title
                fd["author"] = author
                try:
                    if is_cached is True:
                        print(f"Reading {cache_name} from cache")
                        f = open(cache_name)
                    else:
                        f = open(descriptor)
                    dat = f.read(max)
                    self.data += dat
                    fd["data"] = dat
                    fd["index"] = index
                    index += 1
                    self.files.append(fd)
                    f.close()
                    valid = True
                except Exception as e:
                    print(f"ERROR: Cannot read: {cache_name}: {e}")
            if valid is True and is_cached is False and self.cache_dir is not None:
                try:
                    print(f"Caching {cache_name}")
                    f = open(cache_name, 'w')
                    f.write(dat)
                    f.close()
                except Exception as e:
                    print(f"ERROR: failed to save cache {cache_name}: {e}")

        ind = 0
        for c in self.data:  # sets are not deterministic
            if c not in self.c2i:
                self.c2i[c] = ind
                self.i2c[ind] = c
                ind += 1
        self.ptr = 0

    def get_slice(self, length):
        if (self.ptr + length >= len(self.data)):
            self.ptr = 0
        if self.ptr == 0:
            rst = True
        else:
            rst = False
        sl = self.data[self.ptr:self.ptr+length]
        self.ptr += length
        return sl, rst

    def decode(self, ar):
        return ''.join([self.i2c[ic] for ic in ar])

    def encode(self, s):
        return [self.c2i[c] for c in s]

    def get_random_slice(self, length):
        p = random.randrange(0, len(self.data)-length)
        sl = self.data[p:p+length]
        return sl

    def get_slice_array(self, length):
        ar = np.array([c for c in self.get_slice(length)[0]])
        return ar

    def get_encoded_slice(self, length):
        s, rst = self.get_slice(length)
        X = [self.c2i[c] for c in s]
        return X

    def get_encoded_slice_array(self, length):
        return np.array(self.get_encoded_slice(length))

    def get_sample(self, length):
        s, rst = self.get_slice(length+1)
        X = [self.c2i[c] for c in s[:-1]]
        y = [self.c2i[c] for c in s[1:]]
        return (X, y, rst)

    def get_random_sample(self, length):
        s = self.get_random_slice(length+1)
        X = [self.c2i[c] for c in s[:-1]]
        y = [self.c2i[c] for c in s[1:]]
        return (X, y)

    def get_sample_batch(self, batch_size, length):
        smpX = []
        smpy = []
        rst = None
        for i in range(batch_size):
            Xi, yi, rst = self.get_sample(length)
            smpX.append(Xi)
            smpy.append(yi)
        return smpX, smpy, rst

    def get_random_sample_batch(self, batch_size, length):
        smpX = []
        smpy = []
        for i in range(batch_size):
            Xi, yi = self.get_random_sample(length)
            smpX.append(Xi)
            smpy.append(yi)
        return np.array(smpX), np.array(smpy)

    def one_hot(self, p, dim):
        o = np.zeros(p.shape+(dim,), dtype=int)
        for y in range(p.shape[0]):
            for x in range(p.shape[1]):
                o[y, x, p[y, x]] = 1
        return o

    def get_random_onehot_sample_batch(self, batch_size, length):
        X, y = self.get_random_sample_batch(batch_size, length)
        return self.one_hot(X, len(self.i2c)), y

def build(args:dict) -> TextLibrary:
    gbl = GutenbergLib()
    gbl.load_index()
    book_list = gbl.search(args['book_list'])
    libdesc = create_libdesc(gbl, args['project_name'], args['project_description'], book_list=book_list)
    textlib = TextLibrary(libdesc["lib"], text_data_cache_directory=CACHE_DIR)
    return textlib