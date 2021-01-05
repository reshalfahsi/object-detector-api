"""
TextLibrary class. 

Text library for training, encoding, batch generation, and formatted source display. 
It read some books from Project Gutenberg and supports creation of training batches. 
The output functions support highlighting to allow to compare generated texts with the actual sources 
to help to identify identical (memorized) parts of a given length.

Mostly copy-paste from https://github.com/domschl/torch-poet
"""

import os
import random
import numpy as np
from urllib.request import urlopen 

class TextLibrary:
    def __init__(self, descriptors, text_data_cache_directory=None, max=100000000):
        self.descriptors = descriptors
        self.data = ''
        self.cache_dir=text_data_cache_directory
        self.files = []
        self.c2i = {}
        self.i2c = {}
        index = 1
        dat = None
        for descriptor, author, title in descriptors:
            fd = {}
            cache_name = self.get_cache_name(self.cache_dir, author, title)
            if os.path.exists(cache_name):
                is_cached=True
            else:
                is_cached=False
            valid=False
            if descriptor[:4] == 'http' and is_cached is False:
                try:
                    print(f"Downloading {cache_name}")
                    dat = urlopen(descriptor).read().decode('utf-8')
                    if dat[0]=='\ufeff':  # Ignore BOM
                        dat=dat[1:]
                    dat = dat.replace('\r', '')  # get rid of pesky LFs 
                    self.data += dat
                    fd["title"] = title
                    fd["author"] = author
                    fd["data"] = dat
                    fd["index"] = index
                    index += 1
                    valid=True
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
                    valid=True
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
        
    def get_cache_name(self, cache_path, author, title):
        if cache_path is None:
            return None
        cname=f"{author} - {title}.txt"
        cname=cname.replace('?','_')  # Gutenberg index is pre-Unicode-mess and some titles contain '?' for bad conversions.
        cache_filepath=os.path.join(cache_path, cname)
        return cache_filepath

    def display_colored_html(self, textlist, dark_mode=False, display_ref_anchor=True, pre='', post=''):
        bgcolorsWht = ['#d4e6e1', '#d8daef', '#ebdef0', '#eadbd8', '#e2d7d5', '#edebd0',
                    '#ecf3cf', '#d4efdf', '#d0ece7', '#d6eaf8', '#d4e6f1', '#d6dbdf',
                    '#f6ddcc', '#fae5d3', '#fdebd0', '#e5e8e8', '#eaeded', '#A9CCE3']
        bgcolorsDrk = ['#342621','#483a2f', '#3b4e20', '#2a3b48', '#324745', '#3d3b30',
                    '#3c235f', '#443f4f', '#403c37', '#463a28', '#443621', '#364b5f',
                    '#264d4c', '#2a3553', '#3d2b40', '#354838', '#3a3d4d', '#594C23']
        if dark_mode is False:
            bgcolors=bgcolorsWht
        else:
            bgcolors=bgcolorsDrk
        out = ''
        for txt, ind in textlist:
            txt = txt.replace('\n', '<br>')
            if ind == 0:
                out += txt
            else:
                if display_ref_anchor is True:
                    anchor="<sup>[" + str(ind) + "]</sup>"
                else:
                    anchor=""
                out += "<span style=\"background-color:"+bgcolors[ind % 16]+";\">" + \
                       txt + "</span>"+ anchor
        # display(HTML(pre+out+post))

    def source_highlight(self, txt, minQuoteSize=10, dark_mode=False, display_ref_anchor=True):
        tx = txt
        out = []
        qts = []
        txsrc = [("Sources: ", 0)]
        sc = False
        noquote = ''
        while len(tx) > 0:  # search all library files for quote 'txt'
            mxQ = 0
            mxI = 0
            mxN = ''
            found = False
            for f in self.files:  # find longest quote in all texts
                p = minQuoteSize
                if p <= len(tx) and tx[:p] in f["data"]:
                    p = minQuoteSize + 1
                    while p <= len(tx) and tx[:p] in f["data"]:
                        p += 1
                    if p-1 > mxQ:
                        mxQ = p-1
                        mxI = f["index"]
                        mxN = f"{f['author']}: {f['title']}"
                        found = True
            if found:  # save longest quote for colorizing
                if len(noquote) > 0:
                    out.append((noquote, 0))
                    noquote = ''
                out.append((tx[:mxQ], mxI))
                tx = tx[mxQ:]
                if mxI not in qts:  # create a new reference, if first occurence
                    qts.append(mxI)
                    if sc:
                        txsrc.append((", ", 0))
                    sc = True
                    txsrc.append((mxN, mxI))
            else:
                noquote += tx[0]
                tx = tx[1:]
        if len(noquote) > 0:
            out.append((noquote, 0))
            noquote = ''
        self.display_colored_html(out, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor)
        if len(qts) > 0:  # print references, if there is at least one source
            self.display_colored_html(txsrc, dark_mode=dark_mode, display_ref_anchor=display_ref_anchor, pre="<small><p style=\"text-align:right;\">",
                                     post="</p></small>")

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
        o=np.zeros(p.shape+(dim,), dtype=int)
        for y in range(p.shape[0]):
            for x in range(p.shape[1]):
                o[y,x,p[y,x]]=1
        return o
    
    def get_random_onehot_sample_batch(self, batch_size, length):
        X, y = self.get_random_sample_batch(batch_size, length)
        return self.one_hot(X,len(self.i2c)), y