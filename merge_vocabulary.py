# coding=utf-8
# Merge two vocabulary files into one, without duplicating words that are present in both
#
# Usage: python merge_vocabulary.py {vocab1} {vocab2} > {out_file}

import sys

vocabulary = dict()


for line in sys.stdin:
    word, count = line.split(" ")
    count = int(count)
    if word in vocabulary:
        vocabulary[word] += count
    else:
        vocabulary[word] = count

for word in sorted(vocabulary.keys(), key=lambda x: vocabulary[x], reverse=True):
    print("{} {}".format(word, vocabulary[word]))
