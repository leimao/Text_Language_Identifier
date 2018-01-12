import pickle
import numpy as np
import json
from model import *

MODEL_FILENAME = 'saved_model.pkl'
LABEL_ENCODER_FILENAME = 'saved_label_encoder.pkl'
N_GRAMS_FILENAME = 'n_grams.txt'
TEST_DATA_FILENAME = 'test_X_languages_homework.json.txt'
TRAIN_DATA_FILENAME = 'train_X_languages_homework.json.txt'
TRAIN_LABEL_FILENAME = 'train_y_languages_homework.json.txt'
LANGUAGE_CODE_FILENAME = "language-codes_json.json"

def read_language_code(filename):
    textfile = open("language-codes_json.json", "r")
    text = textfile.read()
    language_codes = json.loads(text)

    language_dict = dict()
    for language_code in language_codes:
        language_dict[language_code['alpha2']] = language_code['English']
    return language_dict



def calculate_accuracy(predictions, labels):
    # Compare predictions and labels, and return the prediction accuracy
    # predictions: list
    # labels: list
    
    total = 0
    correct = 0
    for prediction, label in zip(predictions, labels):
        total += 1
        if prediction == label:
            correct += 1
    return float(correct)/total
        

def save_prediction(predictions, filename = 'predictions.txt'):
    # Save the predictions to file
    # predictions: list
    # filename: filename for saved predictions

    with open(filename, 'w') as file:
        for prediction in predictions:
            file.write("{\"classification\":\"%s\"}\n" % prediction)
    return


def load_model(model_filename = MODEL_FILENAME, label_encoder_filename = LABEL_ENCODER_FILENAME, n_grams_filename = N_GRAMS_FILENAME, language_code_filename = LANGUAGE_CODE_FILENAME):
    # Load model from file
    with open(model_filename, 'rb') as file:  
        model = pickle.load(file)

    # Load label encoder from file
    with open(label_encoder_filename, 'rb') as file:  
        le = pickle.load(file)

    n_gram_list = load_n_grams(filename = n_grams_filename)

    lc = read_language_code(filename = language_code_filename)

    lc['lorem'] = 'Lorem Ipsum'

    return model, le, lc, n_gram_list


def classify_text(text, model, le, n_gram_list):

    data_n_gram = prepare_n_gram_dataset(dataset = [text], ns = [1,2,3], n_gram_list = n_gram_list)
    predictions = model.predict(data_n_gram)
    predictions = list(le.inverse_transform(predictions))

    return predictions[0]



if __name__ == '__main__':


    model_filename = MODEL_FILENAME
    label_encoder_filename = LABEL_ENCODER_FILENAME
    test_data_filename = TEST_DATA_FILENAME


    # Load model from file
    with open(model_filename, 'rb') as file:  
        model = pickle.load(file)

    # Load label encoder from file
    with open(label_encoder_filename, 'rb') as file:  
        le = pickle.load(file)

    dataset = read_data(filename = TRAIN_DATA_FILENAME)
    labelset = read_label(filename = TRAIN_LABEL_FILENAME)
    labelset_numeric = le.transform(labelset)

    n_gram_list = load_n_grams(filename = N_GRAMS_FILENAME)
    data_n_gram = prepare_n_gram_dataset(dataset = dataset, ns = [1,2,3], n_gram_list = n_gram_list)

    predictions = model.predict(data_n_gram)
    predictions = list(le.inverse_transform(predictions))



    #print("Metric 1: %f" %model.score(data_n_gram, labelset_numeric))

    print("Prediction Accuracy: %f" %calculate_accuracy(predictions = predictions, labels = labelset))

    #print("Metric 3: %f" %calculate_accuracy(predictions = list(le.inverse_transform(le.transform(labelset))), labels = labelset))

