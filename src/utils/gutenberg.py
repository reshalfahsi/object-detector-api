"""
Gutenberg Library dataset.

Search, filter, clean and download books from Project Gutenberg.

Mostly copy-paste from https://github.com/domschl/torch-poet
"""

import os
import re
import time
import logging
from enum import Enum
from urllib.request import urlopen

CACHE_DIR = "~/.cache/gutenberg"

class GutenbergLib(object):
    """ A fuzzy, lightweight library to access, search and filter Project Gutenberg resources """

    def __init__(self, root_url="http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg", cache_dir=CACHE_DIR):
        """ GutenbergLib by default uses a mirror's root URL

        root_url -- url of Project Gutenberg or any mirror URL.
        cache_dir -- path to a directory that will be used to cache the Gutenberg index and already downloaded texts
        """
        self.log = logging.getLogger('GutenbergLib')
        self.root_url = root_url
        self.index = None
        self.NEAR = 2048
        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.cache_dir = cache_dir
        except Exception as e:
            self.cache_dir = None
            self.log.error(
                f"Failed to create cache directory {cache_dir}, {e}")

    def _parse_record(self, record, verbose=True):
        """ internal function to recreate some consistent record information from near-freestyle text """
        rl = record.split('\n')
        # non-breaking space, TAB, and space
        white = str(chr(160))+str(chr(9))+" "
        ebook_no = ""
        while len(rl[0]) > 0 and rl[0][-1] in white:
            rl[0] = rl[0][:-1]
        while len(rl[0]) > 0 and not rl[0][-1] in white:
            ebook_no = rl[0][-1]+ebook_no
            rl[0] = rl[0][:-1]
        while len(rl[0]) > 0 and rl[0][-1] in white:
            rl[0] = rl[0][:-1]

        # Sanity check
        try:
            fa = re.findall(ebook_no, "\A[0-9]+[A-C]\Z")
        except Exception as e:
            fa = None
            if verbose is True:
                self.log.debug(f"Failed to apply regex on >{ebook_no}<")

        if len(rl[0]) < 5 or fa == None or len(ebook_no) > 7:
            if verbose is True:
                print("-------------------------------------")
                print(record)
                print("- - - - - - - - - - - - - - - - - - -")
                print(f"Dodgy record: {rl[0]}")
                print(f"    ebook-id:  >{ebook_no}<")
            return None

        for i in range(len(rl)):
            rl[i] = rl[i].strip()

        p = 0
        while p < len(rl)-1:
            if len(rl[p+1]) == 0:
                print(f"Invalid rec: {record}")
                p += 1
            else:
                if rl[p+1][0] != "[":
                    rl[p] += " "+rl[p+1]
                    del rl[p+1]
                    if rl[p][-1] == ']':
                        p += 1
                else:
                    p += 1

        rec = {}
        l0 = rl[0].split(", by ")
        rec['title'] = l0[0]
        rec['ebook_id'] = ebook_no
        # if len(l0)>2:
        #    print(f"Chaos title: {rl[0]}")
        if len(l0) > 1:
            rec['author'] = l0[-1]
        for r in rl[1:]:
            if r[0] != '[' or r[-1] != ']':
                if r[0] == '[':
                    ind = r.rfind(']')
                    if ind != -1:
                        # print(f"Garbage trail {r}")
                        r = r[:ind+1]
                        # print(f"Fixed: {r}")
                    else:
                        # print(f"Missing closing ] {r}")
                        r += ']'
                        # print(f"Fixed: {r}")
            if r[0] == '[' and r[-1] == ']':
                r = r[1:-1]
                i1 = r.find(':')
                if i1 == -1:
                    r = r.replace("Author a.k.a.", "Author a.k.a.:")
                    i1 = r.find(':')
                if i1 != -1:
                    i2 = r[i1:].find(' ')+i1
                else:
                    i2 = -1
                if i1 == -1 and i2 == -1:
                    pass
                    # print(f"Invalid attribut in {rl}::{r}")
                else:
                    if i2-i1 == 1:
                        key = r[:i1]
                        val = r[i2+1:]
                        if '[' in key or ']' in key or '[' in val or ']' in val or len(key) > 15:
                            pass
                            # print("messy key/val")
                        else:
                            rec[key.strip().lower()] = val.strip()
                    else:
                        pass
                        # print(f"Bad attribute name terminator, missing ': ' {r}")
            else:
                pass
                # print(f"Invalid attribut in {rl}::{r}")
        if len(rec) > 1:
            if "language" not in rec.keys():
                rec["language"] = "English"
        return rec

    def _parse_index(self, lines):
        """ internal function to parse the fuzzy text-based Gutenberg table of content """
        class State(Enum):
            NONE = 1,
            SYNC_START = 2,
            SYNC_REC = 3,
            END = 5

        # non-breaking space, TAB, and space
        white = str(chr(160))+str(chr(9))+" "
        state = State.NONE
        start_token = "~ ~ ~ ~"
        stop_token = ["====="]
        end_token = "<==End"
        ignore_headers = ["TITLE and AUTHOR"]
        ignore_content = ["Not in the Posted Archives",
                          "human-read audio ebooks", "Audio:"]
        empty_lines = 0
        records = []
        rec = ''
        for line in lines:
            if line[:len(end_token)] == end_token:
                state = State.END
                break

            if state == State.NONE:
                if line[:len(start_token)] == start_token:
                    state = State.SYNC_START
                    empty_lines = 0
                    continue
            if state == State.SYNC_START:
                if len(line.strip()) == 0:
                    empty_lines += 1
                    if empty_lines > 1:
                        state = State.NONE
                        continue
                else:
                    stopped = False
                    for stop in stop_token:
                        if line[:len(stop)] == stop:
                            stopped = True
                            break
                    if stopped is True:
                        state = State.NONE
                        empty_lines = 0
                        continue
                    ignore = False
                    for header in ignore_headers:
                        if line[:len(header)] == header:
                            empty_lines = 0
                            ignore = True
                    for token in ignore_content:
                        if token in line:
                            empty_lines = 0
                            ignore = True
                    if ignore is True:
                        continue
                    rec = line
                    state = State.SYNC_REC
                    continue
            if state == State.SYNC_REC:
                if len(line.strip()) == 0 or line[0] not in white:
                    if len(records) < 10:
                        parsed_rec = self._parse_record(rec, verbose=True)
                    else:
                        parsed_rec = self._parse_record(rec, verbose=False)

                    if parsed_rec is not None:
                        records.append(parsed_rec)
                    empty_lines = 1
                    if len(line.strip()) == 0:
                        state = State.SYNC_START
                        continue
                    else:
                        rec = line
                        continue
                rec = rec + "\n" + line
        return records

    def load_index(self, cache=True, cache_expire_days=30):
        """ This function loads the Gutenberg record index, either from cache, or from a website

        cache -- default True, use the cache directory to cache both index and text files. Index
        expires after cache_expire_days, text files never expire. Should *NOT* be set to False
        in order to prevent unnecessary re-downloading.
        cache_expire_days -- Number of days after which the index is re-downloaded."""
        raw_index = None
        if self.cache_dir is None:
            self.log.error(
                "Cannot cache library index, no valid cache directory.")
            return False
        ts_file = os.path.join(self.cache_dir, "timestamp")
        cache_file = os.path.join(self.cache_dir, "gutenberg_index")
        expired = True
        read_from_cache = False
        if os.path.isfile(ts_file) and os.path.isfile(cache_file):
            try:
                with open(ts_file, 'r') as f:
                    ts = float(f.read())
                if time.time()-ts < cache_expire_days*24*3600:
                    expired = False
                    read_from_cache = True
                    self.log.debug("Cache timestamp read.")
                else:
                    self.log.debug(
                        "Cache for index is expired, reloading from web.")
            except:
                self.log.debug(
                    "Failed to read cache timestamp, reloading from web.")
        if expired is False and os.path.isfile(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    raw_index = f.read()
                    self.log.info(f"Gutenberg index read from {cache_file}")
            except:
                expired = True
                self.log.debug(
                    "Failed to read cached index, reloading from web.")
        if expired is True:
            index_url = self.root_url+"/GUTINDEX.ALL"
            try:
                raw_index = urlopen(index_url).read().decode('utf-8')
                if raw_index[0] == '\ufeff':  # Ignore BOM
                    raw_index = raw_index[1:]
                raw_index = raw_index.replace('\r', '')
                self.log.info(f"Gutenberg index read from {index_url}")
            except Exception as e:
                self.log.error(
                    f"Failed to download Gutenberg index from {index_url}, {e}")
                return False
        if cache is True and read_from_cache is False:
            try:
                with open(ts_file, 'w') as f:
                    f.write(str(time.time()))
                    self.log.debug("Wrote read cache timestamp.")
            except Exception as e:
                print(f"Failed to write cache timestamp to {ts_file}, {e}")
            try:
                with open(cache_file, 'w') as f:
                    f.write(raw_index)
                    self.log.debug("Wrote read cached index.")
            except Exception as e:
                print(f"Failed to write cached index to {cache_file}, {e}")
        lines = raw_index.split('\n')
        self.records = self._parse_index(lines)

    def load_book(self, ebook_id : str) -> str:
        """ get text of an ebook from Gutenberg by ebook_id 

        ebook_id -- Gutenberg id
        """
        file_url = None
        cache_file = None
        if ebook_id is None or len(ebook_id) == 0:
            return None
        if ebook_id[-1] == 'C':
            ebook_id = ebook_id[:-1]
        path_stub = ""

        for i in range(len(ebook_id)-1):
            path_stub += "/"+ebook_id[i]
        path_stub += "/"+ebook_id+"/"
        filenames = [(ebook_id+"-0.txt", 'utf-8'), (ebook_id+".txt", 'utf-8'),
                     (ebook_id+"-8.txt", "latin1"), (ebook_id+".txt", "latin1")]
        cache_name = ebook_id+".txt"
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, cache_name)
            if os.path.isfile(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = f.read()
                        self.log.info(f"Book read from cache at {cache_file}")
                        return data
                except Exception as e:
                    self.log.error(f"Failed to read cached file {cache_file}")
        data = None
        for filename, encoding in filenames:
            file_url = self.root_url+path_stub+filename
            try:
                data = urlopen(file_url).read().decode(encoding)
                self.log.info(f"Book downloaded from {file_url}")
                break
            except Exception as e:
                self.log.debug(f"URL-Download failed: {file_url}, {e}")
                pass
        if data is None:
            self.log.warning(
                f"Failed to download {filenames}, last URL {file_url}, skipping book.")
            return None
        if self.cache_dir is not None:
            try:
                with open(cache_file, 'w') as f:
                    f.write(data)
            except:
                self.log.error(f"Failed to cache file {cache_file}")
        return data

    def filter_text(self, book_text : str) -> str:
        """ Heuristically remove header and trailer texts not part of the actual book 
        """
        start_tokens = ["*** START OF THIS PROJECT", "E-text prepared by",
                        "This book was generously provided by the "]
        near_start_tokens = ["produced by ", "Produced by ", "Transcriber's Note",
                             "Transcriber's note:", "Anmerkungen zur Tanskription"]
        end_tokens = ["End of the Project Gutenberg", "*** END OF THIS PROJECT", "***END OF THE PROJECT GUTENBER",
                      "Ende dieses Projekt Gutenberg", "End of Project Gutenberg", "Transcriber's Note"]
        blen = len(book_text)

        pstart = 0
        for token in start_tokens:
            pos = book_text.find(token)
            if pos > pstart:
                pstart = pos
                self.log.debug(
                    f"Start-token [{token}] found at position {pos}")
        if pstart > 0:
            pos = book_text[pstart:].find("\n\n")
            if pos >= 0 and pos <= self.NEAR:
                pos += pstart
                while book_text[pos] == '\n':
                    pos += 1  # eof?!
                pstart = pos
        if pstart > blen/2:
            self.log.warning("Preamble is taking more than half of the book!")
        new_book = book_text[pstart:]

        xpos = -1
        for token in near_start_tokens:
            pos = new_book.find(token)
            if pos >= 0 and pos <= self.NEAR:
                self.log.debug(
                    f"Near-Start-token [{token}] found at position {pos}")
                if pos > xpos:
                    xpos = pos
        if xpos > -1:
            pos2 = new_book[xpos:].find("\n\n")
            self.log.debug(f"Trying extra skipping for {pos2}...")
            if pos2 <= self.NEAR and pos2 > 0:
                self.log.debug("Trying extra skipping (2)...")
                while new_book[xpos+pos2] == '\n':
                    pos2 += 1
                new_book = new_book[xpos+pos2:]
                self.log.debug(
                    f"Additionally shortened start by {xpos+pos2} chars")

        pend = len(new_book)
        for token in end_tokens:
            pos = new_book.find(token)
            if pos != -1 and pos < pend:
                self.log.debug(f"End-token [{token}] found at pos {pos}")
                pend = pos
        if pend < len(new_book):
            pos = new_book[:pend].rfind("\n\n")
            if pos > 0:
                while new_book[pos] == '\n':
                    pos -= 1  # eof?!
                pend = pos+1
        else:
            self.log.debug("No end token found!")
        if pend < len(new_book)/2:
            self.log.warning("End-text is taking more than half of the book!")
        new_book = new_book[:pend]
        return new_book

    def find_keywords(self, *search_keys):
        """ Search of an arbitrary number of keywords in a book record

        returns -- list of records that contain all keywords in any field. """
        frecs = []
        for rec in self.records:
            found = True
            for sk in search_keys:
                subkey = False
                for key in rec.keys():
                    if sk.lower() in key.lower() or sk.lower() in rec[key].lower():
                        subkey = True
                        break
                if subkey is False:
                    found = False
                    break
            if found is True:
                frecs += [rec]
        return frecs

    def search(self, search_dict : dict) -> list :
        """ Search for book record with key specific key values
        For a list of valid keys, use `get_record_keys()`
        Standard keys are:
        ebook_id, author, language, title
        example: search({"title": ["philosoph","phenomen","physic","hermeneu","logic"], "language":"english"})
        Find all books whose titles contain at least one the keywords, language english. Search keys can either be
        search for a single keyword (e.g. english), or an array of keywords. 
        returns -- list of records """
        frecs = []
        for rec in self.records:
            found = True
            for sk in search_dict:
                if sk not in rec:
                    found = False
                    break
                else:
                    skl = search_dict[sk]
                    if not isinstance(skl, list):
                        skl = [skl]
                    nf = 0
                    for skli in skl:
                        if skli.lower() in rec[sk].lower():
                            nf = nf+1
                    if nf == 0:
                        found = False
                        break
            if found is True:
                frecs += [rec]
        return frecs

    def get_record_keys(self):
        """ Get a list of all keys that are used within records. Standard keys are:
        ebook_id, author, language, title

        returns -- list of all different keys that are somehow used."""
        rks = []
        for r in self.records:
            rks = set(list(rks) + list(r.keys()))
        return rks

    def get_unique_record_values(self, key):
        """ Get a list of all unique values a given keys has for all records.
        get_unique_records_values('language') returns all languages in Gutenberg."""
        uv = []
        if key not in self.get_record_keys():
            print(f"{key} is not a key used in any record!")
            return None
        for r in self.records:
            if key in r:
                uv = set(list(uv)+[r[key]])
        uv = sorted(uv)
        return uv


def get_cache_name(cache_path : str, author : str, title : str) -> str:
    if cache_path is None:
        return None
    cname = f"{author} - {title}.txt"
    # Gutenberg index is pre-Unicode-mess and some titles contain '?' for bad conversions.
    cname = cname.replace('?', '_')
    cache_filepath = os.path.join(cache_path, cname)
    return cache_filepath


def create_libdesc(gbl: GutenbergLib, project_name: str, description: str, cache_path: str = CACHE_DIR, book_list: list = []) -> dict:
    libdesc = {"name": project_name, "description": description, "lib": []}
    if cache_path is None or not os.path.exists(cache_path):
        print(f"A valid cache {cache_path} is needed!")
        return None
    for book_entry in book_list:
        try:
            book_raw_content = gbl.load_book(book_entry['ebook_id'])
        except Exception as e:
            print(f"Failed to download ebook_id {book_entry}, {e}")
            continue
        if book_raw_content is not None:
            try:
                book_text = gbl.filter_text(book_raw_content)
            except Exception as e:
                print(f"Internal error when filtering {book_entry}, {e}")
                continue
            filename = get_cache_name(
                cache_path, book_entry['author'], book_entry['title'])
            try:
                with open(filename, 'w') as f:
                    f.write(book_text)
                    print(f"Cached {filename}")
                    libdesc["lib"].append(
                        (filename, book_entry['author'], book_entry['title']))
            except Exception as e:
                print(f"Failed to cache {filename}", {e})
    return libdesc