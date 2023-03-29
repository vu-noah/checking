# Noah-Manuel Michael
# 24.03.2023
# Advanced NLP Take-Home-Exam
# get a random selection of (di)transitive verbs from VerbNet

import json
import random
from nltk.corpus import verbnet as vn

# Transitive
transitive_verbs = []

for verb_class in vn.classids():
    for frame in vn.frames(verb_class):
        if frame['description']['primary'] == 'Basic Transitive':
            transitive_verbs = transitive_verbs + vn.lemmas(verb_class)

transitive_verbs_random_sample = random.sample(transitive_verbs, k=100)

with open('transitive_verbs.json', 'w') as outfile:
    json.dump(transitive_verbs_random_sample, outfile)

# Ditransitive
ditransitive_verbs = ['give'] + vn.lemmas('give-13.1')

with open('ditransitive_verbs.json', 'w') as outfile:
    json.dump(ditransitive_verbs, outfile)

# Causative-Inchoative
causative_inchoative_verbs = vn.lemmas('break-45.1')

with open('causative_inchoative_verbs.json', 'w') as outfile:
    json.dump(causative_inchoative_verbs, outfile)
