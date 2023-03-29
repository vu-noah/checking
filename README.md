# checking

This repository contains code for creating challenge datasets to test AllenNLP SRL models' behaviors.

Directories:
- /bert_srl: a directory containing the challenge dataset(s) and predictions by the structured-prediction-bert-srl model
- /lexical_resources: files containing different content which is used to build the challenge dataset(s)
- /srl: a directory containing the challenge dataset(s) and predictions by the structured-prediction-srl model

Within /bert_srl and /srl:
- dataset_and_predictions_XXX.json: these files contain the challenge dataset of the specific run; for each test, they contain information about the test and the capability it ought to test, as well as the data itself, and a pretty printed combination of each datapoint with its expected predictions and the actual predictions by the model
- suite_summary_XXX.txt: these files contain the summary of running the test suites, i.e., the failure rates and example failure instances for each test

Files:
- main.py: the main script
- test_runs.txt: a save file in which the number of times the main script has been run is stored
- requirements.txt: package requirements

Execution of the code:
- run main.py from the command line in the following way:
  C:\Users\User\checking>python main.py

The script will proceed to build samples for each proposed test. Some of the samples involve a random selection of words 
or perturbations, which is why portions of the challenge datasets differ from their corresponding portions in other runs.
For assessing the overall performance, you can take the average of the
failure rate over several runs.
Once the samples are created, the script then carries out a number of different tests for evaluating the behavior of the SRL models,
prints the results to the console and saves them to file.

Note: /bert_srl and /srl contain the data for the first 5 runs. If main.py is executed, the new run will be saved to /checking.