# Text_Language_Identifier

Author: Lei Mao


Date: 1/12/2018

## Introduction

This is a text language identifier using multi-class logistic regression algorithm based on n-gram features. It could classify 56 different languages of text with an accuracy of 75%. 

## Dependencies

* Python 3.6
* Numpy
* Scikit-Learn
* io
* json
* pickle
* time

## Issues

Multi-Layer Perceptron and Naive Bayes does not perform significantly better than logistic regression. 

The model could not be improved better probably due to the quality of training data is low. For example, in the training dataset, "advertisement addressed to members of the House of Representatives ." is labeled as "ru"; "\u2022 Florida \u2022" is labeled as "da". These two examples are found in the first twenty training data in the dataset. 

With training data of better quality, I  would expect the performance of the model could be around 85% - 90% with only logistic regression.

## To-Do List

* Map the language label to true language name in English. For example, in the training data, label "ru" should really be "russian".
* Make a graphical user interface using Tkinter that allows user to paste the utf-8 format text to the model and do the classification.