# Noah-Manuel Michael
# 17.03.2023
# Test CheckList library
# Code along to: https://towardsdatascience.com/checklist-behavioral-testing-of-nlp-models-491cf11f0238

from textblob import TextBlob
from checklist.editor import Editor
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT, INV, DIR
import numpy as np

# # get TextBlob as sentiment classifier model
# print(TextBlob('horrible').sentiment[0])

# # Create Data
# editor = Editor()
#
# # with determined altneration of adjectives
# number_of_samples = 2
# label_names = ['good', 'great']
# template = 'Staying at home is not a {positive_adjective} option.'
# # noinspection PyTypeChecker
# random_output_positive_adjective \
#     = editor.template(template, positive_adjective=label_names, nsamples=number_of_samples).data
#
# print(random_output_positive_adjective)
#
# # with masked tokens, no influence on what tokens will be there, but likely fit the context
# masked_word_template = '{mask} is not a {positive_adjective} option.'
# # noinspection PyTypeChecker
# random_output_positive_adjective_with_masked_word \
#     = editor.template(masked_word_template, positive_adjective=label_names, nsamples=number_of_samples).data
#
# print(random_output_positive_adjective_with_masked_word)

# # create more samples
# positive_adjectives = ['good', 'realistic', 'healthy', 'attractive', 'appealing', 'acceptable', 'best', 'feasible',
#                        'easy', 'ideal', 'affordable', 'economical', 'recommended', 'exciting', 'inexpensive', 'obvious',
#                        'great', 'appropriate', 'effective', 'excellent']
#
# negative_adjectives = ['bad', 'unhealthy', 'expensive', 'boring', 'terrible', 'worst', 'unfeasible', 'unappropriate',
#                        'awful', 'time-consuming']
#
# masked_word_template = '{mask} is not a {adjective} option.'
# # noinspection PyTypeChecker
# samples = editor.template(masked_word_template, adjective=positive_adjectives, nsamples=100)
# # noinspection PyTypeChecker
# samples += editor.template(masked_word_template, adjective=negative_adjectives, nsamples=100)
#
# print(samples.data)


# Create function to predict sentiment
def predict_probabilities(utterances: list):
    """
    Compute the sentiment scores for a number of sentences using TextBlob. Transform the sentiment scores (originally,
    they range from -1 for definitely negative to 1 for definitely positive) into probabilities.
    :param utterances: a list of utterances
    :return: np.hstack((positive_probability, negative_probability)): a matrix containing the positive probability and
    the negative probability for each utterance
    """
    positive_probability = np.array([(TextBlob(u).sentiment[0] + 1) / 2.0 for u in utterances]).reshape(-1, 1)
    # reshape reshapes an array (-1 indicates that the number of rows is to be inferred from the length of the array,
    # if sentences contained 3 elements, we could set this number to 3 to get 3 rows, but as we don't know how many
    # sentences will be passed, this defines an unknown dimension; 1 indicates that the array should have one column)
    # print(positive_probability)

    negative_probability = 1 - positive_probability
    # print(negative_probability)

    return np.hstack((negative_probability, positive_probability))


# Create a wrapper that combines the probabilities (confidences) with prediction labels (0 and 1, 0 being equivalent to
# the first column, which is the negative probabilities, i.e., label 0 == negative; and 1 being equivalent to the second
# column, which is the positive probabilities, i.e., label 1 == positive), returns a tuple of two arrays: the first
# array contains the labels, the second array contains the confidences
wrapped_predictions = PredictorWrapper.wrap_softmax(predict_probabilities)

predictions, confidences = wrapped_predictions(['good', 'awesome', 'cool', 'horrible'])
print(predictions)
print(confidences)