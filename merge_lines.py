# coding=utf-8
# When one line in a source dataset is translated as several lines in a target set.
# the source line is preceded by empty lines.
# This script merges these cases to a single line
#
# Usage:
# python merge_lines.py {source_filename} {target_filename}
#
# This overwrites the input files

import sys

source_filename = sys.argv[1]
target_filename = sys.argv[2]

source_examples = []
target_examples = []

with open(source_filename) as source, open(target_filename) as target:
    source_line = source.readline()
    target_line = target.readline()
    lines_read = 1
    while source_line != "" or target_line != "":
        if source_line == "\n" or target_line == "\n":
            source_line = ((source_line[:-1] + " ") if source_line != "\n" else "") + source.readline()
            target_line = ((target_line[:-1] + " ") if target_line != "\n" else "") + target.readline()
        else:
            source_examples.append(source_line)
            target_examples.append(target_line)
            source_line = source.readline()
            target_line = target.readline()
        lines_read += 1

assert len(source_examples) == len(target_examples)
assert source_line == "" and target_line == ""

print("Reduced {} lines to {} examples".format(lines_read, len(source_examples)))

with open(source_filename, "w") as source:
    source.writelines(source_examples)

with open(target_filename, "w") as target:
    target.writelines(target_examples)
