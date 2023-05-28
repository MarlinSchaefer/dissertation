# Machine Learning Applications in Search Algorithms for Gravitational Waves from Compact Binary Mergers

This repository contains the source code for my thesis.

## Running scripts
All scripts were written in Python 3 and are expected to be executed
from the folder they are stored in. To install dependencies, simply
```
pip install --upgrade pip setuptools wheel
```
and then (in the main directory of this thesis)
```
pip install -r requirements.txt
```

To check for duplicate entries in the bibliography, head to the directory
`/bib/scripts` and start a python environment. Then type
```
from bibparse import *
ct = find_close_titles('../bibliography.bib')
{key: val for (key, val) in ct.items() if len(val) > 0}
```
The resulting printed dictionary will give you a list of titles that are
very similar. Each key-value pair is a title and a list of similar
titles.

## Compiling the document
The document was compiled with `pdfTeX 3.14159265-2.6-1.40.18 (TeX Live 2017/Debian)`.
The full compilation command order for Texmaker is listed in a comment
at the top of main.tex.