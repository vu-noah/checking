from allennlp_models import pretrained

model_srl = pretrained.load_predictor('structured-prediction-srl')
model_bert_srl = pretrained.load_predictor('structured-prediction-srl-bert')

predictions_srl = model_srl.predict('Frederick killed Joan.')
predictions_bert_srl = model_bert_srl.predict('Joan was killed by Frederick.')

print(predictions_srl)
print(predictions_bert_srl)
