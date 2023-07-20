import pandas as pd
import numpy as np
import pickle
from transformers import pipeline
import scispacy
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration

# function for applying the prompt constraint
def apply_prompt(text):
    return 'Context: ' + text + '\nIs it likely the patient has kidney failure?"\nConstraint: Even if you are uncertain, you must pick either “Yes” or “No” without using any other words.'

def check_token(token):
    if token != 'Yes' and token != 'No':
        print('generated weird token:', token)

def document_level_classifier(text, tokenizer, model):

    input_ids = tokenizer(apply_prompt(text), return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=99999)

    prediction = tokenizer.decode(outputs[0])[6:-4]

    check_token(prediction)

    if prediction == 'Yes':
        return 1
    else:
        return 0

def sentence_level_classifier(sentences, tokenizer, model):

    for sentence in sentences:
        input_ids = tokenizer(apply_prompt(sentence), return_tensors="pt").input_ids.to("cuda")

        outputs = model.generate(input_ids, max_new_tokens=99999)

        prediction = tokenizer.decode(outputs[0])[6:-4]

        check_token(prediction[0]['generated_text'])
        if prediction[0]['generated_text'] == 'Yes':
            return 1

    return 0

data = pd.read_csv('csv_files/master_data.csv')

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

# apply the constraint and also ask the model to evaluate yes or no

# this list stores the predictions
predictions = []

cnt = 0

for index, row in data.iterrows():
    #############################
    # the following code is for document level classification
    #############################

    predictions.append(document_level_classifier(row['notes_half1'], tokenizer, model))

    #############################
    # the following code is for sentence level classification
    #############################

    # predictions.append(sentence_level_classifier(sentences, classifier))

    # cnt += 1

    # if cnt == 5:
    #     break

prediction_data = pd.DataFrame({'prediction': predictions})
prediction_data.to_csv('csv_files/predictions-xxl-kidney-fail.csv', index=False)
