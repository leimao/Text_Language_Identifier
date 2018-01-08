import numpy as np
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import time


TRAIN_DATA_RAW = "train_X_languages_homework.json.txt"
TRAIN_LABEL_RAW = "train_y_languages_homework.json.txt"


def read_data(path):
    sentences = list()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sentence = json.loads(line)['text']
                sentences.append(sentence)
        return sentences

def read_label(path):
    classifications = list()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                classification = json.loads(line)['classification']
                classifications.append(classification)
        return classifications


def count_language(labelset):
    label_count = dict()
    for label in labelset:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count


def count_char_utf(dataset):
    char_count = dict()
    for data in dataset:
        for char in data:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
    return char_count


def extract_n_gram(dataset, labelset, language, n = 2, k = 200):
    # extract the top k frequent n-grams for certain language
    # count all n-grams
    n_gram_count = dict()
    for data, label in zip(dataset, labelset):
        if label == language:
            # print(label)
            length = len(data)
            for i in range(length-n+1):
                n_gram = data[i:i+n]
                #print(n_gram)
                if n_gram in n_gram_count:
                    n_gram_count[n_gram] += 1
                else:
                    n_gram_count[n_gram] = 1
    # extract the top k frequent n-grams from all the n-grams
    n_gram_count_tuple = list()
    for n_gram in n_gram_count:
        n_gram_count_tuple.append((n_gram, n_gram_count[n_gram]))
    n_gram_count_tuple.sort(key = lambda tup: tup[1], reverse = True)
    number_n_gram = len(n_gram_count_tuple)
    n_gram_top_k = list()
    n_gram_top_k_occurrence = list()
    for i in range(min(k, number_n_gram)):
        n_gram_top_k.append(n_gram_count_tuple[i][0])
        n_gram_top_k_occurrence.append(n_gram_count_tuple[i][1])
    return n_gram_top_k, n_gram_top_k_occurrence


def extract_n_gram_all(dataset, labelset, languages, n = 2, k = 200):
    # extract the top k frequent n-grams for all languages
    # make them into on n_gram list
    n_gram_list = list()
    n_gram_occurrence = list()
    for language in languages:
        n_gram_top_k, n_gram_top_k_occurrence = extract_n_gram(
            dataset = dataset, labelset = labelset, language = language, n = n, k = k)
        n_gram_list += n_gram_top_k
        
    n_gram_list = list(set(n_gram_list))
    
    return n_gram_list


def n_gram_representation(sentence, ns, n_gram_list):
    # ns is the list of n

    sentence_n_grams = list()
    length = len(sentence)

    for n in ns:
        for i in range(length-n+1):
            #print(sentence[i:i+n])
            sentence_n_grams.append(sentence[i:i+n])

    sentence_n_grams = set(sentence_n_grams)
    
    num_n_grams_all = len(n_gram_list)

    representation = np.zeros(num_n_grams_all)
    
    for i in range(num_n_grams_all):
        if n_gram_list[i] in sentence_n_grams:
            
            representation[i] += 1
    
    return representation

def prepare_n_gram_dataset(dataset, ns, n_gram_list):
    n_gram_dataset = np.zeros((len(dataset), len(n_gram_list)))
    size_dataset = len(dataset)
    for i in range(size_dataset):
        #print(i)
        sentence = dataset[i]
        n_gram_dataset[i] = n_gram_representation(sentence, ns, n_gram_list)
        
    return n_gram_dataset


if __name__ == '__main__':

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Local current time :" + localtime)

    dataset = read_data(path = TRAIN_DATA_RAW)
    labelset = read_label(path = TRAIN_LABEL_RAW)

    data_train, data_test, label_train, label_test = train_test_split(dataset, labelset, test_size = 0.1, random_state = 0)
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size = 0.1, random_state = 0)

    label_count_train = count_language(labelset = label_train)
    print(len(label_count_train))

    char_count_train = count_char_utf(dataset = data_train)
    print(len(char_count_train))

    two_gram_list = extract_n_gram_all(dataset = data_train, labelset = label_train, languages = list(label_count_train.keys()), n = 2, k = 200)
    print("Number of 2-grams: %d" %len(two_gram_list))

    three_gram_list = extract_n_gram_all(dataset = data_train, labelset = label_train, languages = list(label_count_train.keys()), n = 3, k = 50)
    print("Number of 3-grams: %d" %len(three_gram_list))


    #one_gram_list = extract_n_gram_all(dataset = data_train, labelset = label_train, languages = list(label_count_train.keys()), n = 1, k = 50)


    #data_n_gram_train = prepare_n_gram_dataset(dataset = data_train, ns = [2,3], n_gram_list = two_gram_list + three_gram_list)
    #data_n_gram_val = prepare_n_gram_dataset(dataset = data_val, ns = [2,3], n_gram_list = two_gram_list + three_gram_list)
    #data_n_gram_test = prepare_n_gram_dataset(dataset = data_test, ns = [2,3], n_gram_list = two_gram_list + three_gram_list)


    data_n_gram_train = prepare_n_gram_dataset(dataset = data_train, ns = [2], n_gram_list = two_gram_list)
    data_n_gram_val = prepare_n_gram_dataset(dataset = data_val, ns = [2], n_gram_list = two_gram_list)
    data_n_gram_test = prepare_n_gram_dataset(dataset = data_test, ns = [2], n_gram_list = two_gram_list)



    # One-hot encoding labels
    lb = preprocessing.LabelBinarizer()
    lb.fit(label_train)
    #print(lb.classes_)

    label_onehot_train = lb.transform(label_train)
    label_onehot_val = lb.transform(label_val)
    label_onehot_test = lb.transform(label_test)


    # Numeric encoding labels
    le = preprocessing.LabelEncoder()
    le.fit(label_train)
    #print(le.classes_)

    label_numeric_train = le.transform(label_train)
    label_numeric_val = le.transform(label_val)
    label_numeric_test = le.transform(label_test)


    print("Start Logistic Regression")

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Local current time :" + localtime)


    logreg = linear_model.LogisticRegression(solver = 'lbfgs', n_jobs = 12, max_iter = 10)

    for i in range(40):

        print("Training Round: %d" % i)

        localtime = time.asctime( time.localtime(time.time()) )
        print ("Local current time :" + localtime)

        model = logreg.fit(data_n_gram_train, label_numeric_train)

        score_train = model.score(data_n_gram_train, label_numeric_train)
        score_val = model.score(data_n_gram_val, label_numeric_val)
        score_test = model.score(data_n_gram_test, label_numeric_test)

        print("------------------------")

        print("Train Score: %f" %score_train)
        print("Validation Score: %f" %score_val)
        print("Test Score: %f" %score_test)

        print("------------------------")

    localtime = time.asctime( time.localtime(time.time()) )
    print ("Local current time :" + localtime)




