import xml.etree.ElementTree as ElementTree
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]


with open(out_file, mode='w', encoding='utf-8') as fh:
    root = ElementTree.parse(in_file).getroot()[0]
    for doc in root.findall('doc'):
        for element in doc.findall('seg'):
            fh.write(element.text.strip() + '\n')
