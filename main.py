# Noah-Manuel Michael
# 23.03.2023
# Advanced NLP Take-Home-Exam
# Create a challenge dataset and test AllenNLP models

import re
import json
from checklist.editor import Editor
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from allennlp_models import pretrained
from nltk.corpus import verbnet as vn


def get_predictions_srl(sentences):
    """

    :param sentences:
    :return:
    """
    all_predictions = []
    for sentence in sentences:
        all_prediction_info = model_srl.predict(sentence)
        if all_prediction_info['verbs']:
            for verb in all_prediction_info['verbs']:
                predictions = verb['tags']
                all_predictions.append(predictions)
        else:
            all_predictions.append('No_prediction')

    return all_predictions


def get_predictions_bert_srl(sentences):
    """

    :param sentences:
    :return:
    """
    all_predictions = []
    for sentence in sentences:
        all_prediction_info = model_bert_srl.predict(sentence)
        if all_prediction_info['verbs']:
            for verb in all_prediction_info['verbs']:
                predictions = verb['tags']
                all_predictions.append(predictions)
        else:
            all_predictions.append('No_prediction')

    return all_predictions


def write_dataset_and_predictions_to_json(test):
    """

    :param test:
    :return:
    """
    dataset = {'test_name': test.name, 'capability': test.capability, 'description': test.description,
               'templates': test.templates, 'data': test.data,
               'data, expectation, prediction': [tup for tup in zip(test.data, test.labels, test.results['preds'])]}

    with open('dataset.json', 'a') as outfile:
        json.dump(dataset, outfile, indent=4)
        outfile.write('\n')


# Instantiate the editor to create test instances
editor = Editor()

# Load adjectives and nouns to use with editor
# Adjectives taken from: https://gist.github.com/hugsy/8910dc78d208e40de42deb29e62df913; random sample
# Nouns taken from: https://github.com/edthrn/most-common-english-words; random sample
# Transitive verbs obtained from VerbNet; random sample

with open('unambiguous_nouns.json') as infile:
    nouns = json.load(infile)

with open('unambiguous_adjectives.json') as infile:
    adjectives = json.load(infile)

with open('transitive_verbs.json') as infile:
    transitive_verbs = json.load(infile)

with open('ditransitive_verbs.json') as infile:
    ditransitive_verbs = json.load(infile)

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
times = ['midnight', 'noon']
cities = [c for c in editor.lexicons['city'] if all([' ' not in c, '-' not in c])]
countries = [c for c in editor.lexicons['country'] if all([' ' not in c, '-' not in c])]
clocktimes = ['three o\'clock', 'six o\'clock', 'nine o\'clock', 'twelve o\'clock'] + \
             [str(i) + 'pm' for i in range(1, 12)] + [str(i) + 'am' for i in range(1, 12)]

# Instantiate test suites to store the created tests
suite = TestSuite()

# Test 1a ##############################################################################################################
# all_verb_lemmas = vn.lemmas()
# infinitives = [l for l in all_verb_lemmas if re.match(r'^[a-z]+$', l)]  # I do not want lemmas with underscores
#
# samples_t1a = infinitives
#
# test_1a = MFT(data=samples_t1a, labels=[['B-V'] for _ in range(len(samples_t1a))], name='Test_1a',
#               capability='Voc+PoS (C)', description='Recognize predicates.', templates='{verb}')
#
# suite.add(test_1a)

# Test 1b ##############################################################################################################
# infinitives = ['to ' + l for l in all_verb_lemmas if re.match(r'^[a-z]+$', l)]
# samples_t1b = infinitives
#
# test_1b = MFT(data=samples_t1b, labels=[['O', 'B-V'] for _ in range(len(samples_t1b))], name='Test_1b',
#               capability='Voc+PoS (C)', description='Recognize predicates.', templates='to {verb}')
#
# suite.add(test_1b)

# Test 2 ###############################################################################################################
# template_t2 = 'The {noun} exists.'
# samples_t2 = editor.template(template_t2, noun=list(nouns), product=True, remove_duplicates=True,
#                              labels=['B-ARG1', 'I-ARG1', 'B-V', 'O'])
#
# test_2 = MFT(data=samples_t2.data, labels=samples_t2.labels, name='Test_2', capability='Voc+PoS (C)',
#              description='Recognize noun phrases as participants.', templates=template_t2)
#
# suite.add(test_2)

# Test 3 ###############################################################################################################
# template_t3 = 'They {transitive_verb} it.'
# samples_t3 = editor.template(template_t3, transitive_verb=transitive_verbs, product=True, remove_duplicates=True,
#                              labels=['B-ARG0', 'B-V', 'B-ARG1', 'O'])
#
# test_3 = MFT(data=samples_t3.data, labels=samples_t3.labels, name='Test_3', capability='Voc+PoS (C)',
#              description='Recognize transitive predicates.', templates=template_t3)
#
# suite.add(test_3)

# Test 4 ###############################################################################################################
# template_t4_1 = 'They {ditransitive_verb} it to {male}.'
# samples_t4 = editor.template(template_t4_1, ditransitive_verb=ditransitive_verbs, nsamples=50, remove_duplicates=True,
#                              labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARG2', 'I-ARG2', 'O'])
#
# template_t4_2 = 'They {ditransitive_verb} it to {female}.'
# samples_t4 += editor.template(template_t4_2, ditransitive_verb=ditransitive_verbs, nsamples=50, remove_duplicates=True,
#                               labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARG2', 'I-ARG2', 'O'])
#
# test_4 = MFT(data=samples_t4.data, labels=samples_t4.labels, name='Test_4', capability='Voc+PoS (C)',
#              description='Recognize ditransitive predicates.', templates=[template_t4_1, template_t4_2])
#
# suite.add(test_4)

# Test 5 ###############################################################################################################
# template_t5_1 = '{noun}'
# template_t5_2 = '{male}'
# template_t5_3 = '{female}'
# template_t5_4 = '{noun} {noun}'
# template_t5_5 = '{male} {noun}'
# template_t5_6 = '{female} {noun}'
# template_t5_7 = '{noun} {male}'
# template_t5_8 = '{noun} {female}'
# template_t5_9 = '{male} {male}'
# template_t5_10 = '{female} {female}'
#
# samples_t5 = editor.template(template_t5_1, noun=nouns, product=True, remove_duplicates=True, labels='No_prediction')
# samples_t5 += editor.template(template_t5_2, noun=nouns, product=True, remove_duplicates=True, labels='No_prediction')
# samples_t5 += editor.template(template_t5_3, noun=nouns, product=True, remove_duplicates=True, labels='No_prediction')
#
# for template in [template_t5_4, template_t5_5, template_t5_6, template_t5_7, template_t5_8, template_t5_9,
#                  template_t5_10]:
#     samples_t5 += editor.template(template, noun=nouns, nsamples=100, remove_duplicates=True, labels='No_prediction')
#
# test_5 = MFT(data=samples_t5.data, labels=samples_t5.labels, name='Test_5', capability='Voc+PoS (C)',
#              description='Label roles only when predicate exists.',
#              templates=[template_t5_1, template_t5_2, template_t5_3, template_t5_4, template_t5_5, template_t5_6,
#                         template_t5_7, template_t5_8, template_t5_9, template_t5_10])
#
# suite.add(test_5)

# Test 6 ###############################################################################################################
# template_t6_1 = 'He killed her in {city} on {weekday}.'
# template_t6_2 = 'He killed her in {country} on {weekday}.'
# template_t6_3 = 'He killed her in {city} at {time}.'
# template_t6_4 = 'He killed her in {country} at {time}.'
# template_t6_5 = 'He killed her at {clocktime}.'
#
# samples_t6 = editor.template(template_t6_1, weekday=weekdays, city=cities, nsamples=100, remove_duplicates=True,
#                              labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-TMP', 'I-ARGM-TMP',
#                                      'O'])
# samples_t6 += editor.template(template_t6_2, weekday=weekdays, country=countries, nsamples=100, remove_duplicates=True,
#                               labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-TMP', 'I-ARGM-TMP',
#                                       'O'])
# samples_t6 += editor.template(template_t6_3, time=times, city=cities, nsamples=100, remove_duplicates=True,
#                               labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-TMP',
#                                       'I-ARGM-TMP', 'O'])
# samples_t6 += editor.template(template_t6_4, time=times, country=countries, nsamples=100,
#                               remove_duplicates=True, labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-LOC', 'I-ARGM-LOC',
#                                                               'B-ARGM-TMP', 'I-ARGM-TMP', 'O'])
# samples_t6 += editor.template(template_t6_5, clocktime=clocktimes, product=True, remove_duplicates=True,
#                               labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'O'])
#
# test_6 = MFT(data=samples_t6.data, labels=samples_t6.labels, name='Test_6', capability='NER (C)',
#              description='Recognize locations & temporal expressions.',
#              templates=[template_t6_1, template_t6_2, template_t6_3, template_t6_4, template_t6_5])
#
# suite.add(test_6)

# Test 7 ###############################################################################################################
# template_t7_1 = 'He killed her on {weekday} in {city}.'
# template_t7_2 = 'He killed her on {weekday} in {country}.'
# template_t7_3 = 'He killed her at {time} in {city}.'
# template_t7_4 = 'He killed her at {time} in {country}.'
#
# samples_t7 = editor.template(template_t7_1, weekday=weekdays, city=cities, nsamples=100, remove_duplicates=True,
#                              labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC',
#                                      'O'])
# samples_t7 += editor.template(template_t7_2, weekday=weekdays, country=countries, nsamples=100, remove_duplicates=True,
#                               labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC',
#                                       'O'])
# samples_t7 += editor.template(template_t7_3, time=times, city=cities, nsamples=100, remove_duplicates=True,
#                               labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC',
#                                       'O'])
# samples_t7 += editor.template(template_t7_4, time=times, country=countries, nsamples=100,
#                               remove_duplicates=True, labels=['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP',
#                                                               'B-ARGM-LOC', 'I-ARGM-LOC', 'O'])
#
# test_7 = MFT(data=samples_t7.data, labels=samples_t7.labels, name='Test_7', capability='NER (R)',
#              description='Label LOC & TMP correctly if in wrong order.',
#              templates=[template_t7_1, template_t7_2, template_t7_3, template_t7_4])
#
# suite.add(test_7)

# Test 8 ###############################################################################################################
template_t8_1 = 'On {weekday} in {city}, he killed her.'
template_t8_2 = 'On {weekday} in {country}, he killed her.'
template_t8_3 = 'At {time} in {city}, he killed her.'
template_t8_4 = 'At {time} in {country}, he killed her.'

samples_t8 = editor.template(template_t8_1, weekday=weekdays, city=cities, nsamples=100, remove_duplicates=True,
                             labels=['B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'O', 'B-ARG0', 'B-V',
                                     'B-ARG1', 'O'])
samples_t8 += editor.template(template_t8_2, weekday=weekdays, country=countries, nsamples=100, remove_duplicates=True,
                              labels=['B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'O', 'B-ARG0', 'B-V',
                                      'B-ARG1', 'O'])
samples_t8 += editor.template(template_t8_3, time=times, city=cities, nsamples=100, remove_duplicates=True,
                              labels=['B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'O', 'B-ARG0', 'B-V',
                                      'B-ARG1', 'O'])
samples_t8 += editor.template(template_t8_4, time=times, country=countries, nsamples=100, remove_duplicates=True,
                              labels=['B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'O', 'B-ARG0', 'B-V',
                                      'B-ARG1', 'O'])

test_8 = MFT(data=samples_t8.data, labels=samples_t8.labels, name='Test_8', capability='NER (R)',
             description='Label LOC & TMP correctly if at the beginning of the sentence.',
             templates=[template_t8_1, template_t8_2, template_t8_3, template_t8_4])

suite.add(test_8)

# Test 9 ###############################################################################################################
template_t9_1 = '{male} killed her.'
template_t9_2 = ''


test_9 = MFT(data=samples_t9.data, labels=samples_t9.labels, name='Test_9', capability='Semantics (C)',
             description='Distinguish animate and volitional from inanimate and non-volitional participants.',
             templates=template_t9)


# Run the tests ########################################################################################################
# Load models
model_srl = pretrained.load_predictor('structured-prediction-srl')
# model_bert_srl = pretrained.load_predictor('structured-prediction-srl-bert')
#
# # Wrap model predictions to a dummy confidence score of 1.0
wrapped_model_srl = PredictorWrapper.wrap_predict(get_predictions_srl)
# wrapped_model_bert_srl = PredictorWrapper.wrap_predict(get_predictions_bert_srl)
#
# Run test suite
suite.run(wrapped_model_srl, verbose=True)
suite.summary()
#
# suite.run(wrapped_model_bert_srl)
# suite.summary()

# test_1a, test_1b, test_2, test_3, test_4, test_5, test_6, test_7, test_8
for test in [test_8]:
    write_dataset_and_predictions_to_json(test)
