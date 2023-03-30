# Noah-Manuel Michael
# 24.03.2023
# Advanced NLP Take-Home-Exam
# get a random selection of nouns and make sure they are unambiguous with spaCy

import random
import spacy
import json

nlp = spacy.load('en_core_web_sm')

nouns = set()
with open('nouns.txt') as infile:
    for line in infile.readlines():
        nouns.add(line.strip('\n'))

unambiguous_nouns = []

i = 0
while i < 100:
    sample_word = random.sample(nouns, 1)[0]
    doc = nlp(sample_word)
    for token in doc:
        if token.pos_ == 'NOUN' and token.text not in unambiguous_nouns:
            unambiguous_nouns.append(token.text)
            i += 1

with open('unambiguous_nouns.json', 'w') as outfile:
    json.dump(unambiguous_nouns, outfile)
