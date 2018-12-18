# coding=utf-8
# Discard tag lines

# Usage:
# python discard_tags.py {source_filename} {target_filename}

import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file) as fh1:
    with open(out_file, "w") as fh2:
        for line in fh1:
            if line.startswith("<"):
                continue
            fh2.write(line)

