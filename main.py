# Noah-Manuel Michael
# 23.03.2023
# Advanced NLP Take-Home-Exam
# Create a challenge dataset and test AllenNLP models

import re
import json
import numpy
import os
import sys
from checklist.editor import Editor
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from allennlp_models import pretrained
from nltk.corpus import verbnet as vn
from checklist.perturb import Perturb


def get_predictions_srl(sentences):
    """
    Extract the predictions from an AllenNLP (srl) model. Combine predictions if sentence has more than one verb.
    :param list[str] sentences: the input sentences
    :return: list[str] all_predictions: a list of strings of the predictions
    """
    all_predictions = []
    for sentence in sentences:
        all_prediction_info = model_srl.predict(sentence)
        if all_prediction_info['verbs']:
            if len(all_prediction_info['verbs']) > 1:
                prediction_list = ['O'
                                   for _
                                   in range(len(all_prediction_info['verbs'][0]['tags']))
                                   ]
                for verb in all_prediction_info['verbs']:
                    for i, prediction in enumerate(verb['tags']):
                        if prediction != 'O':
                            prediction_list[i] = prediction
                all_predictions.append(str(prediction_list))
            else:
                predictions = all_prediction_info['verbs'][0]['tags']
                all_predictions.append(str(predictions))
        else:
            all_predictions.append('No_prediction')

    return all_predictions


def get_predictions_bert_srl(sentences):
    """
    Extract the predictions from an AllenNLP (bert-srl) model. Combine predictions if sentence has more than one verb.
    :param list[str] sentences: the input sentences
    :return: list[str] all_predictions: a list of strings of the predictions
    """
    all_predictions = []
    for sentence in sentences:
        all_prediction_info = model_bert_srl.predict(sentence)
        if all_prediction_info['verbs']:
            if len(all_prediction_info['verbs']) > 1:
                prediction_list = ['O'
                                   for _
                                   in range(len(all_prediction_info['verbs'][0]['tags']))
                                   ]
                for verb in all_prediction_info['verbs']:
                    for i, prediction in enumerate(verb['tags']):
                        if prediction != 'O':
                            prediction_list[i] = prediction
                all_predictions.append(str(prediction_list))
            else:
                predictions = all_prediction_info['verbs'][0]['tags']
                all_predictions.append(str(predictions))
        else:
            all_predictions.append('No_prediction')

    return all_predictions


def write_dataset_and_predictions_to_json(test, model, run):
    """
    Extract metadata from tests and write dataset to file.
    :param test: a CheckList test object with results
    :param str model: the model used for predictions
    :param int run: the number of the test run
    :return: None
    """
    if type(test.results['preds'][0]) == numpy.ndarray:  # for DIR and INV
        dataset = {'test_name': test.name,
                   'capability': test.capability,
                   'description': test.description,
                   'templates': test.templates,
                   'data': test.data,
                   'data, expectation, prediction': [tup for tup
                                                     in zip(test.data, test.labels, [array.tolist()
                                                                                     for array
                                                                                     in test.results['preds']
                                                                                     ])]
                   }
    else:  # for MFT
        dataset = {'test_name': test.name,
                   'capability': test.capability,
                   'description': test.description,
                   'templates': test.templates,
                   'data': test.data,
                   'data, expectation, prediction': [tup for tup
                                                     in zip(test.data, test.labels, test.results['preds'])]
                   }

    if model == 'model_srl':
        with open(f'dataset_and_predictions_srl_{run}.json', 'a') as outfile:
            json.dump(dataset, outfile, indent=4)
            outfile.write('\n')
    elif model == 'model_bert_srl':
        with open(f'dataset_and_predictions_bert_srl_{run}.json', 'a') as outfile:
            json.dump(dataset, outfile, indent=4)
            outfile.write('\n')


# Instantiate the editor to create test instances
editor = Editor()
# Instantiate test suites to store the created tests
suite_srl = TestSuite()
suite_bert_srl = TestSuite()


# Load adjectives and nouns to use with editor
# Nouns taken from: https://github.com/edthrn/most-common-english-words; random sample
# Transitive verbs obtained from VerbNet; random sample
# Tools taken from: https://preply.com/en/blog/names-of-tools-in-english/ and https://7esl.com/tools-vocabulary/
# Ambiguous names taken from: https://www.madeformums.com/pregnancy/place-name-baby-names/; sample

with open('lexical_resources/unambiguous_nouns.json') as infile:
    nouns = json.load(infile)

with open('lexical_resources/transitive_verbs.json') as infile:
    transitive_verbs = json.load(infile)

with open('lexical_resources/ditransitive_verbs.json') as infile:
    ditransitive_verbs = json.load(infile)

with open('lexical_resources/causative_inchoative_verbs.json') as infile:
    causative_inchoative_verbs = json.load(infile)


weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

times = ['midnight', 'noon']

tools = ['screw', 'nut', 'handsaw', 'bradawl', 'bolt', 'hammer', 'screwdriver', 'mallet', 'axe', 'saw', 'scissors',
         'chisel', 'pliers', 'drill', 'nail', 'wrench', 'backsaw', 'hacksaw', 'pocketknife', 'chainsaw', 'stone',
         'brace', 'corkscrew', 'plunger', 'stepladder']

ambiguous_names = ['Adelaide', 'Arizona', 'Asia', 'Austin', 'Avalon', 'Boston', 'Bristol', 'Brooklyn', 'Camden',
                   'Carolina', 'Charlotte', 'Chelsea', 'Chester', 'Dakota', 'Dallas', 'Denver', 'Devon', 'Florence',
                   'Holland', 'India', 'Ireland', 'Italia', 'Jackson', 'Jordan', 'Kent', 'Kenya', 'Kingston', 'Lille',
                   'Milan', 'Odessa', 'Orlando', 'Paris', 'Phoenix', 'Raleigh', 'Regina', 'Rhodes', 'Rio', 'Rochelle',
                   'Rome', 'Sahara', 'Savannah', 'Siena', 'Sorrento', 'Sydney', 'Valencia', 'Vienna']


clocktimes = ['three o\'clock', 'six o\'clock', 'nine o\'clock', 'twelve o\'clock'] + \
             [str(i) + 'pm'
              for i
              in range(1, 12)
              ] + [str(i) + 'am'
                   for i
                   in range(1, 12)
                   ]


male_names = [n for n
              in editor.lexicons['male']]
female_names = [n for n
                in editor.lexicons['female']]
cities = [c for c
          in editor.lexicons['city']
          if all([' ' not in c,
                  '-' not in c])]
countries = [c for c
             in editor.lexicons['country']
             if all([' ' not in c,
                     '-' not in c])]


# Test 1a ##############################################################################################################
all_verb_lemmas = vn.lemmas()

infinitives = [l for l
               in all_verb_lemmas
               if re.match(r'^[a-z]+$', l)]  # I do not want lemmas with underscores

samples_t1a = infinitives

test_1a = MFT(data=samples_t1a,
              labels=[str(['B-V'])
                      for _ in range(len(samples_t1a))],
              name='Test 1a',
              capability='Voc+PoS (C)',
              description='Recognize predicates.',
              templates='{verb}',
              )

suite_srl.add(test_1a)
suite_bert_srl.add(test_1a)

# Test 1b ##############################################################################################################
infinitives = ['to ' + l
               for l
               in all_verb_lemmas
               if re.match(r'^[a-z]+$', l)]

samples_t1b = infinitives

test_1b = MFT(data=samples_t1b,
              labels=[str(['O', 'B-V'])
                      for _
                      in range(len(samples_t1b))],
              name='Test 1b',
              capability='Voc+PoS (C)',
              description='Recognize predicates.',
              templates='to {verb}',
              )

suite_srl.add(test_1b)
suite_bert_srl.add(test_1b)

# Test 2 ###############################################################################################################
template_t2 = 'The {noun} exists.'
samples_t2 = editor.template(template_t2, noun=list(nouns), product=True, remove_duplicates=True,
                             labels=str(['B-ARG1', 'I-ARG1', 'B-V', 'O']))

test_2 = MFT(data=samples_t2.data,
             labels=samples_t2.labels,
             name='Test 2',
             capability='Voc+PoS (C)',
             description='Recognize noun phrases as participants.',
             templates=template_t2,
             )

suite_srl.add(test_2)
suite_bert_srl.add(test_2)

# Test 3 ###############################################################################################################
template_t3 = 'They {transitive_verb} it.'

expectation_t3 = str(['B-ARG0', 'B-V', 'B-ARG1', 'O'])

samples_t3 = editor.template(template_t3, transitive_verb=transitive_verbs, product=True, remove_duplicates=True,
                             labels=expectation_t3)

test_3 = MFT(data=samples_t3.data,
             labels=samples_t3.labels,
             name='Test 3',
             capability='Voc+PoS (C)',
             description='Recognize transitive predicates.',
             templates=template_t3,
             )

suite_srl.add(test_3)
suite_bert_srl.add(test_3)

# Test 4 ###############################################################################################################
template_t4_1 = 'They {ditransitive_verb} it to {male}.'
template_t4_2 = 'They {ditransitive_verb} it to {female}.'

expectation_t4 = str(['B-ARG0', 'B-V', 'B-ARG1', 'B-ARG2', 'I-ARG2', 'O'])

samples_t4 = editor.template(template_t4_1, ditransitive_verb=ditransitive_verbs, nsamples=50, remove_duplicates=True,
                             labels=expectation_t4)
samples_t4 += editor.template(template_t4_2, ditransitive_verb=ditransitive_verbs, nsamples=50, remove_duplicates=True,
                              labels=expectation_t4)

test_4 = MFT(data=samples_t4.data,
             labels=samples_t4.labels,
             name='Test 4',
             capability='Voc+PoS (C)',
             description='Recognize ditransitive predicates.',
             templates=[template_t4_1, template_t4_2],
             )

suite_srl.add(test_4)
suite_bert_srl.add(test_4)

# Test 5 ###############################################################################################################
template_t5_1 = '{noun}'
template_t5_2 = '{male}'
template_t5_3 = '{female}'
template_t5_4 = '{noun} {noun}'
template_t5_5 = '{male} {noun}'
template_t5_6 = '{female} {noun}'
template_t5_7 = '{noun} {male}'
template_t5_8 = '{noun} {female}'
template_t5_9 = '{male} {male}'
template_t5_10 = '{female} {female}'

samples_t5 = editor.template(template_t5_1, noun=nouns, product=True, remove_duplicates=True, labels='No_prediction')
samples_t5 += editor.template(template_t5_2, noun=nouns, product=True, remove_duplicates=True, labels='No_prediction')
samples_t5 += editor.template(template_t5_3, noun=nouns, product=True, remove_duplicates=True, labels='No_prediction')

for template in [template_t5_4, template_t5_5, template_t5_6, template_t5_7, template_t5_8, template_t5_9,
                 template_t5_10]:
    samples_t5 += editor.template(template, noun=nouns, nsamples=100, remove_duplicates=True, labels='No_prediction')

test_5 = MFT(data=samples_t5.data,
             labels=samples_t5.labels,
             name='Test 5',
             capability='Voc+PoS (C)',
             description='Label roles only when predicate exists.',
             templates=[template_t5_1, template_t5_2, template_t5_3, template_t5_4, template_t5_5, template_t5_6,
                        template_t5_7, template_t5_8, template_t5_9, template_t5_10],
             )

suite_srl.add(test_5)
suite_bert_srl.add(test_5)

# Test 6 ###############################################################################################################
template_t6_1 = 'He killed her in {city} on {weekday}.'
template_t6_2 = 'He killed her in {country} on {weekday}.'
template_t6_3 = 'He killed her in {city} at {time}.'
template_t6_4 = 'He killed her in {country} at {time}.'
template_t6_5 = 'He killed her at {clocktime}.'

expectation_t6_1 = str(['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-TMP', 'I-ARGM-TMP', 'O'])
expectation_t6_2 = str(['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'O'])

samples_t6 = editor.template(template_t6_1, weekday=weekdays, city=cities, nsamples=100, remove_duplicates=True,
                             labels=expectation_t6_1)
samples_t6 += editor.template(template_t6_2, weekday=weekdays, country=countries, nsamples=100, remove_duplicates=True,
                              labels=expectation_t6_1)
samples_t6 += editor.template(template_t6_3, time=times, city=cities, nsamples=100, remove_duplicates=True,
                              labels=expectation_t6_1)
samples_t6 += editor.template(template_t6_4, time=times, country=countries, nsamples=100, remove_duplicates=True,
                              labels=expectation_t6_1)
samples_t6 += editor.template(template_t6_5, clocktime=clocktimes, product=True, remove_duplicates=True,
                              labels=expectation_t6_2)

test_6 = MFT(data=samples_t6.data,
             labels=samples_t6.labels,
             name='Test 6',
             capability='NER (C)',
             description='Recognize locations & temporal expressions.',
             templates=[template_t6_1, template_t6_2, template_t6_3, template_t6_4, template_t6_5],
             )

suite_srl.add(test_6)
suite_bert_srl.add(test_6)

# Test 7 ###############################################################################################################
template_t7_1 = 'He killed her on {weekday} in {city}.'
template_t7_2 = 'He killed her on {weekday} in {country}.'
template_t7_3 = 'He killed her at {time} in {city}.'
template_t7_4 = 'He killed her at {time} in {country}.'

expectation_t7 = str(['B-ARG0', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'O'])

samples_t7 = editor.template(template_t7_1, weekday=weekdays, city=cities, nsamples=100, remove_duplicates=True,
                             labels=expectation_t7)
samples_t7 += editor.template(template_t7_2, weekday=weekdays, country=countries, nsamples=100, remove_duplicates=True,
                              labels=expectation_t7)
samples_t7 += editor.template(template_t7_3, time=times, city=cities, nsamples=100, remove_duplicates=True,
                              labels=expectation_t7)
samples_t7 += editor.template(template_t7_4, time=times, country=countries, nsamples=100, remove_duplicates=True,
                              labels=expectation_t7)

test_7 = MFT(data=samples_t7.data,
             labels=samples_t7.labels,
             name='Test 7',
             capability='NER (R)',
             description='Label LOC & TMP correctly if in wrong order.',
             templates=[template_t7_1, template_t7_2, template_t7_3, template_t7_4],
             )

suite_srl.add(test_7)
suite_bert_srl.add(test_7)

# Test 8 ###############################################################################################################
template_t8_1 = 'On {weekday} in {city}, he killed her.'
template_t8_2 = 'On {weekday} in {country}, he killed her.'
template_t8_3 = 'At {time} in {city}, he killed her.'
template_t8_4 = 'At {time} in {country}, he killed her.'

expectation_t8 = str(['B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'O', 'B-ARG0', 'B-V', 'B-ARG1', 'O'])

samples_t8 = editor.template(template_t8_1, weekday=weekdays, city=cities, nsamples=100, remove_duplicates=True,
                             labels=expectation_t8)
samples_t8 += editor.template(template_t8_2, weekday=weekdays, country=countries, nsamples=100, remove_duplicates=True,
                              labels=expectation_t8)
samples_t8 += editor.template(template_t8_3, time=times, city=cities, nsamples=100, remove_duplicates=True,
                              labels=expectation_t8)
samples_t8 += editor.template(template_t8_4, time=times, country=countries, nsamples=100, remove_duplicates=True,
                              labels=expectation_t8)

test_8 = MFT(data=samples_t8.data,
             labels=samples_t8.labels,
             name='Test 8',
             capability='NER (R)',
             description='Label LOC & TMP correctly if at the beginning of the sentence.',
             templates=[template_t8_1, template_t8_2, template_t8_3, template_t8_4],
             )

suite_srl.add(test_8)
suite_bert_srl.add(test_8)

# Test 9 ###############################################################################################################
template_t9_animate = '{male} killed her.'
template_t9_inanimate = 'The {tool} killed her.'

samples_t9_animate = editor.template(template_t9_animate, nsamples=25, remove_duplicates=True,
                                     labels=str(['B-ARG0', 'B-V', 'B-ARG1', 'O']))
samples_t9_inanimate = editor.template(template_t9_inanimate, tool=tools, product=True, remove_duplicates=True,
                                       labels=str(['B-ARG2', 'I-ARG2', 'B-V', 'B-ARG1', 'O']))

animate_inanimate_data = [[animate, inanimate]
                          for animate, inanimate
                          in zip(samples_t9_animate.data, samples_t9_inanimate.data)]


def different_SRs_for_inanimate(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    """
    Expectation function for animate vs. inanimate distinction. Parameters are filled automatically.
    :param orig_pred: the predictions for the first sentence
    :param pred: the predictions for the second sentence
    :param orig_conf: /
    :param conf: /
    :param labels: /
    :param meta: /
    :return: bool: whether the labels for both sentences were predicted correctly
    """
    if all([str(orig_pred.tolist()) == str(['B-ARG0', 'B-V', 'B-ARG1', 'O']),
            str(pred.tolist()) == str(['B-ARG2', 'I-ARG2', 'B-V', 'B-ARG1', 'O'])]):
        return True
    else:
        return False


expect_function_animate_inanimate = Expect.pairwise(different_SRs_for_inanimate)

test_9 = DIR(data=animate_inanimate_data,
             expect=expect_function_animate_inanimate,
             name='Test 9',
             capability='Semantics (C)',
             description='Distinguish animate and volitional from inanimate and non-volitional participants.',
             templates=[template_t9_animate, template_t9_inanimate],
             labels=[[l1, l2]
                     for l1, l2
                     in zip(samples_t9_animate.labels, samples_t9_inanimate.labels)],
             )

suite_srl.add(test_9)
suite_bert_srl.add(test_9)

# Test 11 ##############################################################################################################
template_t11_active = '{male} killed {female}.'
template_t11_passive = '{female} was killed by {male}.'

expectation_t11_1 = str(['B-ARG0', 'B-V', 'B-ARG1', 'O'])
expectation_t11_2 = str(['B-ARG1', 'B-V', 'B-V', 'B-ARG0', 'I-ARG0', 'O'])

active_sentences = []
passive_sentences = []

for male_name, female_name in zip(male_names, female_names):
    active_sentences.append(f'{male_name} killed {female_name}.')
    passive_sentences.append(f'{female_name} was killed by {male_name}.')

active_passive_data = [[active, passive]
                       for active, passive
                       in zip(active_sentences, passive_sentences)]


def active_passive_shift(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    """
    Expectation function for active-passive alternation. Parameters are filled automatically.
    :param orig_pred: the predictions for the first sentence
    :param pred: the predictions for the second sentence
    :param orig_conf: /
    :param conf: /
    :param labels: /
    :param meta: /
    :return: bool: whether the labels for both sentences were predicted correctly
    """
    if all([str(orig_pred.tolist()) == expectation_t11_1,
            str(pred.tolist()) == expectation_t11_2]):
        return True
    else:
        return False

expect_function_active_passive = Expect.pairwise(active_passive_shift)

test_11 = DIR(active_passive_data,
              expect=expect_function_active_passive,
              name='Test 11',
              capability='Alternation (C)',
              description='Handle active-passive alternation correctly.',
              templates=[template_t11_active, template_t11_passive],
              labels=[[expectation_t11_1, expectation_t11_2]
                      for _
                      in range(100)],
              )

suite_srl.add(test_11)
suite_bert_srl.add(test_11)

# Test 12 ##############################################################################################################
template_t12_PP = 'They {ditransitive_verb} the money to her.'
template_t12_dative = 'They {ditransitive_verb} her the money.'

expectation_t12_1 = str(['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'O'])
expectation_t12_2 = str(['B-ARG0', 'B-V', 'B-ARG2', 'B-ARG1', 'I-ARG1', 'O'])

PP_sentences = []
dative_sentences = []

for ditransitive_verb in ditransitive_verbs:
    PP_sentences.append(f'They {ditransitive_verb} the money to her.')
    dative_sentences.append(f'They {ditransitive_verb} her the money.')

PP_dative_data = [[PP, dative]
                  for PP, dative
                  in zip(PP_sentences, dative_sentences)]


def PP_dative_shift(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    """
    Expectation function for PP-dative alternation. Parameters are filled automatically.
    :param orig_pred: the predictions for the first sentence
    :param pred: the predictions for the second sentence
    :param orig_conf: /
    :param conf: /
    :param labels: /
    :param meta: /
    :return: bool: whether the labels for both sentences were predicted correctly
    """
    if all([str(orig_pred.tolist()) == expectation_t12_1,
            str(pred.tolist()) == expectation_t12_2]):
        return True
    else:
        return False


expect_function_PP_dative = Expect.pairwise(PP_dative_shift)

test_12 = DIR(PP_dative_data,
              expect=expect_function_PP_dative,
              name='Test 12',
              capability='Alternation (C)',
              description='Handle alternation of dative-like construction and PP properly.',
              templates=[template_t12_PP, template_t12_dative],
              labels=[[expectation_t12_1, expectation_t12_2]
                      for _ in range(8)],
              )

suite_srl.add(test_12)
suite_bert_srl.add(test_12)

# Test 13 ##############################################################################################################
template_t13_causative = 'He will {causative_inchoative_verb} the window.'
template_t13_inchoative = 'The window will {causative_inchoative_verb}.'

expectation_t13_1 = str(['B-ARG0', 'B-ARGM-MOD', 'B-V', 'B-ARG1', 'I-ARG1', 'O'])
expectation_t13_2 = str(['B-ARG1', 'I-ARG1', 'B-ARGM-MOD', 'B-V', 'O'])

causative_sentences = []
inchoative_sentences = []

for causative_inchoative_verb in causative_inchoative_verbs:
    causative_sentences.append(f'He will {causative_inchoative_verb} the window.')
    inchoative_sentences.append(f'The window will {causative_inchoative_verb}.')

causative_inchoative_data = [[causative, inchoative]
                             for causative, inchoative
                             in zip(causative_sentences, inchoative_sentences)]


def causative_inchoative_shift(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    """
    Expectation function for causative-inchoative alternation. Parameters are filled automatically.
    :param orig_pred: the predictions for the first sentence
    :param pred: the predictions for the second sentence
    :param orig_conf: /
    :param conf: /
    :param labels: /
    :param meta: /
    :return: bool: whether the labels for both sentences were predicted correctly
    """
    if all([str(orig_pred.tolist()) == expectation_t13_1,
            str(pred.tolist()) == expectation_t13_2]):
        return True
    else:
        return False


expect_function_causative_inchoative = Expect.pairwise(causative_inchoative_shift)

test_13 = DIR(causative_inchoative_data,
              expect=expect_function_causative_inchoative,
              name='Test 13',
              capability='Alternation (C)',
              description='Handle causative-inchoative alternation correctly.',
              templates=[template_t13_causative, template_t13_inchoative],
              labels=[[expectation_t13_1, expectation_t13_2]
                      for _ in range(13)],
              )

suite_srl.add(test_13)
suite_bert_srl.add(test_13)

# Test 14 ##############################################################################################################
template_t14 = 'They the {noun} {transitive_verb}.'

samples_t14 = editor.template(templates=template_t14, transitive_verb=transitive_verbs, noun=nouns,
                              labels=str(['B-ARG0', 'B-ARG1', 'I-ARG1', 'B-V', 'O']),
                              nsamples=200, remove_duplicates=True)

test_14 = MFT(data=samples_t14.data,
              labels=samples_t14.labels,
              name='Test 14',
              capability='Word Order (R)',
              description='Resolve SRL even when word order is incorrect.',
              templates=template_t14,
              )

suite_srl.add(test_14)
suite_bert_srl.add(test_14)

# Test 16 ##############################################################################################################
samples_t16 = samples_t3 + samples_t4

typo_sentences = Perturb.perturb(samples_t16.data, Perturb.add_typos)

test_16 = INV(data=typo_sentences.data,
              name='Test 16',
              capability='Robustness (R)',
              description='Spelling mistakes should not affect the predictions.',
              templates=[template_t3, template_t4_1, template_t4_2],
              labels=[[expectation_t3, expectation_t3]  # labels are not expected to change pairwise
                      for _ in range(100)]  # first half of samples
                     +
                     [[expectation_t4, expectation_t4]
                      for _ in range(100)]  # second half
              )

suite_srl.add(test_16)
suite_bert_srl.add(test_16)

# Test 18 ##############################################################################################################
template_t18 = '{ambiguous_name} touches the dog.'

samples_t18 = editor.template(templates=template_t18, ambiguous_name=ambiguous_names,
                              labels=str(['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'O']),
                              product=True, remove_duplicates=True)

test_18 = MFT(data=samples_t18.data,
              labels=samples_t18.labels,
              name='Test 17',
              capability='Ambiguity (C)',
              description='Handle ambiguity based on semantic criteria (of the predicate).',
              templates=template_t18,
              )

suite_srl.add(test_18)
suite_bert_srl.add(test_18)

# Run the tests ########################################################################################################
# Load models
model_srl = pretrained.load_predictor('structured-prediction-srl')
model_bert_srl = pretrained.load_predictor('structured-prediction-srl-bert')

# Wrap model predictions to a dummy confidence score of 1.0
wrapped_model_srl = PredictorWrapper.wrap_predict(get_predictions_srl)
wrapped_model_bert_srl = PredictorWrapper.wrap_predict(get_predictions_bert_srl)

# Check which run of the script it is
if 'test_runs.txt' in os.listdir():
    with open('test_runs.txt') as infile:
        for line in infile.readlines():
            run = int(line.strip())
        run = run + 1
else:
    run = 1

with open('test_runs.txt', 'a') as outfile:
    outfile.write(str(run) + '\n')

# Run test suite_srl
print('Testing srl model.')
suite_srl.run(wrapped_model_srl, verbose=True)
suite_srl.summary()

with open(f'suite_summary_srl_{run}.txt', 'w') as outfile:
    sys.stdout = outfile
    suite_srl.summary()
    sys.stdout = sys.__stdout__

for test in [test_1a, test_1b, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_11, test_12,
             test_13, test_14, test_16, test_18]:
    write_dataset_and_predictions_to_json(test, 'model_srl', run)

# Run test suite_bert_srl
print('Testing bert srl model.')
suite_bert_srl.run(wrapped_model_bert_srl, verbose=True, overwrite=True)
suite_bert_srl.summary()

with open(f'suite_summary_bert_srl_{run}.txt', 'w') as outfile:
    sys.stdout = outfile
    suite_bert_srl.summary()
    sys.stdout = sys.__stdout__

for test in [test_1a, test_1b, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_11, test_12,
             test_13, test_14, test_16, test_18]:
    write_dataset_and_predictions_to_json(test, 'model_bert_srl', run)
