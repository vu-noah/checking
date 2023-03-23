from checklist.editor import Editor
from checklist.pred_wrapper import PredictorWrapper
from checklist.test_types import MFT
from allennlp_models import pretrained

editor = Editor()

test_template = '{male_name} killed {female_name}.'
samples = editor.template(test_template, male_name=['John', 'Louis'], female_name=['Mary', 'Eva'], product=True,
                          labels=['B-ARG0', 'B-V', 'B-ARG1', 'O'], remove_duplicates=True)

print(samples.data)

model_bert_srl = pretrained.load_predictor('structured-prediction-srl-bert')

def make_predictions(input_sequences):

    all_predictions = []
    for input_sequence in input_sequences:
        all_prediction_info = model_bert_srl.predict(input_sequence)
        predictions = all_prediction_info['verbs'][0]['tags']
        all_predictions.append(predictions)

    return all_predictions

wrapped_model = PredictorWrapper.wrap_predict(make_predictions)

minimum_functionality_test = MFT(data=samples.data, labels=samples.labels, templates=test_template,
                                 name='Test simple SRL', capability='NER',
                                 description='Test whether the sentiment model is able to handle people names '
                                             'appropriately.')

minimum_functionality_test.run(wrapped_model)
print(minimum_functionality_test.name)
print(minimum_functionality_test.capability)
print(minimum_functionality_test.description)
print(minimum_functionality_test.templates)
print(minimum_functionality_test.data)
print(minimum_functionality_test.labels)
print(minimum_functionality_test.results)
minimum_functionality_test.summary()
