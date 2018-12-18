# coding=utf-8
# Extract sentences from a .sgm file into one segment per line
#
# Usage:
# python extract_sgm.py {input_file}
#
# Writes the output to a file with the same name as the input but without .sgm extension

import sys
from bs4 import BeautifulSoup

in_filename = sys.argv[1]
out_filename = in_filename[:-4]  # without .sgm extension

with open(in_filename) as fh:
    soup = BeautifulSoup(fh)

with open(out_filename, "w") as fh:
    for doc in soup.html.body.srcset.find_all("doc"):
        for seg in doc.find_all("seg"):
            fh.write(seg.string + "\n")
