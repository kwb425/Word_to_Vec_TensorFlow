#######################################################################################################################
# Data organizer:
#                 KakaoTalk -> CSV
#                 -> MS Excel (or TextWrangler) -> UTF-16 (one sentence per one line format without blank lines)
#                 -> THIS SCRIPT -> UTF-8 (one word per one line format)
#
#                                                                                               Written by Kim, Wiback,
#                                                                                                     2017.05.11. v1.1.
#######################################################################################################################
import codecs
import os
from nltk.tokenize import word_tokenize





## Organizing #########################################################################################################



######
# Path
######
DATADIR = os.path.join(os.path.dirname(__file__), os.path.pardir, "data")
INPUT = os.path.join(DATADIR, "crude.txt")
OUTPUT = os.path.join(DATADIR, "input")



######
# Main
######

### Read
fid = codecs.open(INPUT, "r", encoding="utf-16")
words = [word_tokenize(sentence) for sentence in fid.readlines()]
fid.close()

### Organizing (one word per one line format)
fid = codecs.open(OUTPUT, "w", encoding="utf-8")
for i in range(len(words)):
    for ii in range(len(words[i])):
        fid.write(u"{}\n".format(words[i][ii]))
fid.close()
