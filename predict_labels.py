import pandas as pd
import numpy as np
import pickle
from transformers import pipeline
import scispacy
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration

# function for applying the prompt constraint
def apply_prompt(text):
    return 'Context: ' + text + '\nIs it likely the patient has atrial fibrillation?"\nConstraint: Even if you are uncertain, you must pick either “Yes” or “No” without using any other words.'

def check_token(token):
    if token != 'Yes' and token != 'No':
        print('generated weird token:', token)

def document_level_classifier(text, classifier):

    # document = ''
    # for sentence in sentences:
    #     document += sentence
    prediction = classifier(apply_prompt(text))

    check_token(prediction[0]['generated_text'])

    if prediction[0]['generated_text'] == 'Yes':
        return 1
    else:
        return 0

def sentence_level_classifier(sentences, classifier):
    for sentence in sentences:
        prediction = classifier(apply_prompt(sentence))
        check_token(prediction[0]['generated_text'])
        if prediction[0]['generated_text'] == 'Yes':
            return 1

    return 0

def sentence_level_classifier_half(sentences, classifier):
    yes_label = 0
    for sentence in sentences:
        prediction = classifier(apply_prompt(sentence))
        check_token(prediction[0]['generated_text'])
        if prediction[0]['generated_text'] == 'Yes':
            yes_label += 1

    if yes_label > (len(sentences) // 2):
        return 1
    else:
        return 0

data = pd.read_csv('csv_files/processed_afib_data.csv')

# notes_half_1_sentences = pickle.load(open('list_of_sentences.p', 'rb'))

classifier = pipeline(model='google/flan-t5-xxl', device_map="auto")
# classifier = pipeline(model='google/flan-t5-xxl')

# apply the constraint and also ask the model to evaluate yes or no

# this list stores the predictions
predictions = []

cnt = 0

for index, row in data.iterrows():
    #############################
    # the following code is for document level classification
    #############################

    predictions.append(document_level_classifier(row['notes_half2'], classifier))

    #############################
    # the following code is for sentence level classification
    #############################

    # predictions.append(sentence_level_classifier(sentences, classifier))

    #############################
    # the following code is for sentence level classification, but more than half the labels need to be yes
    #############################

    # predictions.append(sentence_level_classifier_half(sentences, classifier))

    # cnt += 1

    # if cnt == 5:
    #     break

prediction_data = pd.DataFrame({'prediction': predictions})
prediction_data.to_csv('csv_files/predictions-xxl-document-half2.csv', index=False)
