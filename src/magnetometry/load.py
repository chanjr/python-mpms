#!/usr/bin/env python3

import os
import re


def natural_sort(l):
    convert = lambda s: int(s) if s.isdigit() else s
    digit_split = re.compile(r"(\d+)")
    key = lambda x: tuple(convert(c) for c in digit_split.split(x))
    return sorted(l, key=key)
    l.sort(key=key)


def load_dir(directory, extension="", cls=None, numerical_sort=True):
    data = []
    for root, dirs, files in os.walk(directory):
        files_ext = sorted(
            [os.path.join(root, f) for f in files if f.lower().endswith(extension)]
        )
        if numerical_sort:
            files_ext = natural_sort(files_ext)
        if cls is not None:
            data += [cls(f) for f in files_ext]
        else:
            data += files_ext
    return data


def load(fnames=None, extension="", cls=None, numerical_sort=True):
    if not fnames:
        fnames = ["."]
    data = []
    for fname in fnames:
        if os.path.isdir(fname):
            data += load_dir(
                fname, extension=extension, cls=cls, numerical_sort=numerical_sort
            )
        else:
            if cls is not None:
                data.append(cls(fname))
            else:
                data.append(fname)
    return data
