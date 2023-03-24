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


# def write_dataset_and_predictions_to_json(results):
#     store = {}
#     for result in results:
#         store['data'] = results.data
#         store['expectation'] = expectations
#         store['prediction_model1'] = model_prediction
#         store['prediction_model2'] = model_prediction2


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


# Extract predictions from models ######################################################################################
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


# Instantiate test suites to store the created tests
suite = TestSuite()

# Instantiate the editor to create test instances
editor = Editor()

# Test 1a ##############################################################################################################
# # Samples
# all_verb_lemmas = vn.lemmas()
# infinitives = [l for l in all_verb_lemmas if re.match(r'^[a-z]+$', l)]
#
# samples_t1a = infinitives
#
# # Test
# test_1a = MFT(data=samples_t1a, labels=[['B-V'] for _ in range(len(samples_t1a))], name='Test_1a',
#               capability='Voc+PoS', description='Recognize predicates.', templates='{verb}')
#
# suite.add(test_1a)

# Test 1b ##############################################################################################################
# # Samples
# infinitives = ['to ' + l for l in all_verb_lemmas if re.match(r'^[a-z]+$', l)]
# samples_t1b = infinitives
#
# # Test
# test_1b = MFT(data=samples_t1b, labels=[['O', 'B-V'] for _ in range(len(samples_t1b))], name='Test_1b',
#               capability='Voc+PoS', description='Recognize predicates.', templates='to {verb}')
#
# suite.add(test_1b)

# Test 2 ###############################################################################################################
# # Samples
# template_t2 = 'The {noun} exists.'
# samples_t2 = editor.template(template_t2, noun=list(nouns), product=True, remove_duplicates=True,
#                              labels=['B-ARG1', 'I-ARG1', 'B-V', 'O'])
#
# # Test
# test_2 = MFT(data=samples_t2.data, labels=samples_t2.labels, name='Test_2', capability='Voc+PoS',
#              description='Recognize noun phrases as participants.', templates=template_t2)
#
# suite.add(test_2)

# Test 3 ###############################################################################################################
template_t3 = 'They {transitive_verb} it.'




print(transitive_verbs)

quit()

# Run the tests ########################################################################################################
# # Load models
model_srl = pretrained.load_predictor('structured-prediction-srl')
# model_bert_srl = pretrained.load_predictor('structured-prediction-srl-bert')
#
# # Wrap model predictions to a dummy confidence score of 1.0
wrapped_model_srl = PredictorWrapper.wrap_predict(get_predictions_srl)
# wrapped_model_bert_srl = PredictorWrapper.wrap_predict(get_predictions_bert_srl)
#
# # Run test suite
suite.run(wrapped_model_srl)
suite.summary()
#
# suite.run(wrapped_model_bert_srl)
# suite.summary()

