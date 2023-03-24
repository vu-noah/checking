# Noah-Manuel Michael
# 24.03.2023
# Advanced NLP Take-Home-Exam
# get a random selection of transitive verbs from VerbNet

import json
import random
from nltk.corpus import verbnet as vn

transitive_verbs = []

for verb_class in vn.classids():
    for frame in vn.frames(verb_class):
        if frame['description']['primary'] == 'Basic Transitive':
            transitive_verbs = transitive_verbs + vn.lemmas(verb_class)

transitive_verbs_random_sample = random.sample(transitive_verbs, 100)

with open('transitive_verbs.json', 'w') as outfile:
    json.dump(transitive_verbs, outfile)
