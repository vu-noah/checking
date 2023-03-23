# Noah-Manuel Michael
# 23.03.2023
# Advanced NLP Take-Home-Exam
# Create a challenge dataset and test AllenNLP models

import re
from checklist.editor import Editor
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from allennlp_models import pretrained
from nltk.corpus import verbnet


# def write_dataset_and_predictions_to_json(results):
#     store = {}
#     for result in results:
#         store['data'] = results.data
#         store['expectation'] = expectations
#         store['prediction_model1'] = model_prediction
#         store['prediction_model2'] = model_prediction2


# Load adjectives.txt and nouns.txt to use with editor
# Adjectives taken from: https://gist.github.com/hugsy/8910dc78d208e40de42deb29e62df913
# Nouns taken from:


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

    # all_predictions = []
    # for sentence in sentences:
    #     all_prediction_info = model_bert_srl.predict(sentence)
    #     predictions = all_prediction_info['verbs'][0]['tags']
    #     all_predictions.append(predictions)
    #
    # return all_predictions


# Instantiate test suites to store the created tests
suite = TestSuite()

# Instantiate the editor to create test instances
editor = Editor()

# Test 1a ##############################################################################################################
# # Samples
# all_verb_lemmas = verbnet.lemmas()
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
#
# samples_t1b = infinitives
#
# # Test
# test_1b = MFT(data=samples_t1b, labels=[['O', 'B-V'] for _ in range(len(samples_t1b))], name='Test_1b',
#               capability='Voc+PoS', description='Recognize predicates.', templates='to {verb}')
#
# suite.add(test_1b)

# Test 2 ###############################################################################################################
# Samples
nouns = set()
nouns.update(set(editor.suggest('This is a good {mask}.')), set(editor.suggest('This is a bad {mask}.')),
             set(editor.suggest('This is a great {mask}.')), set(editor.suggest('This is a horrible {mask}.')),
             set(editor.suggest('This is a fast {mask}.')), set(editor.suggest('This is a funny {mask}.')),
             set(editor.suggest('This is a beautiful {mask}.')), set(editor.suggest('This is a distracting {mask}.')),
             set(editor.suggest('This is a male {mask}.')), set(editor.suggest('This is a female {mask}.')))
print(nouns)

adjectives = set()
adjectives.update(set(editor.suggest('This is a {mask} child.')), set(editor.suggest('This is a {mask} memory.')),
             set(editor.suggest('This is a {mask} castle.')), set(editor.suggest('This is a {mask} person.')),
             set(editor.suggest('This is a {mask} experiment.')), set(editor.suggest('This is a {mask} movie.')),
             set(editor.suggest('This is a {mask} job.')), set(editor.suggest('This is a {mask} stone.')),
             set(editor.suggest('This is a {mask} cat.')), set(editor.suggest('This is a {mask} headline.')))
print(adjectives)

name_list = editor.lexicons['male']

# template_t2 = '{mask} exists.'
#
# samples_t2 = editor.template(template_t2, nsamples=200, remove_duplicates=True, labels=['B-ARG0', 'B-V'])
#
# print(samples_t2.data)





# Run the tests ########################################################################################################
# # Load models
# model_srl = pretrained.load_predictor('structured-prediction-srl')
# model_bert_srl = pretrained.load_predictor('structured-prediction-srl-bert')
#
# # Wrap model predictions to a dummy confidence score of 1.0
# wrapped_model_srl = PredictorWrapper.wrap_predict(get_predictions_srl)
# wrapped_model_bert_srl = PredictorWrapper.wrap_predict(get_predictions_bert_srl)
#
# # Run test suite
# suite.run(wrapped_model_srl)
# suite.summary()
#
# suite.run(wrapped_model_bert_srl)
# suite.summary()

