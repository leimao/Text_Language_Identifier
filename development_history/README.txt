Language detection


Attached is a langauge dataset fabricated from different language wikipedias. The dataset is not manicured. We have including a training dataset with X and y labels, and a testing file without y labels to submit. You job is to create a model that correctly predicts a language for a given text snippet. There is no time limit given, but we ask that you don't spent more than the equivalent of a few evenings out of respect for your time.


Your completed solution should be a zipfile with the following:

    1) build_model script: should read train_X_languages_homework.json.txt and train_y_languages_homework.json.txt to build, train and tune your best predictive model. The script should save your best model to the filesystem and it should also log data about the expected performance of
    the model to a text file performance.txt

    2) make_predictions script: should load your saved model generated by the first script, read the file test_X_languages_homework.json.txt
    and output a predictions.txt file with your predictions (one prediction per line).

    3) model binary: serialized version of your best model from running step 1

    4) predictions.txt: one prediction per line that is in the same order as test_X_languages_homework.json.txt

    5) performance.txt: the data about the expected performance of your model, and how you evaluated the best model.

    6) notes.txt: should include how long you spent on the assignment, the software dependencies for your code, notes about how you chose your model, 
    how you engineered your features, what other options you wanted to explore but didn't have time.



Feel free to use whatever standard machine learning packages you'd like.

Do not use additional data from the internet. However you can derive new features from the existing data.

The homework will be judged based upon the performance of the model, the quality/clarity of the code, and the
methodology of the best feature/model selection.
