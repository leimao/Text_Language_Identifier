
# coding: utf-8

# In[34]:


import pickle
import numpy as np
import json
from model import *


# In[35]:


MODEL_FILENAME = 'saved_model.pkl'
LABEL_ENCODER_FILENAME = 'saved_label_encoder.pkl'
TEST_DATA_FILENAME = 'test_X_languages_homework.json.txt'
TRAIN_DATA_FILENAME = 'train_X_languages_homework.json.txt'


# In[36]:


test_data = read_data(path = TRAIN_DATA_FILENAME)


# In[37]:


test_data = test_data[0:2000]


# In[38]:


model_filename = MODEL_FILENAME
label_encoder_filename = LABEL_ENCODER_FILENAME
test_data_filename = TEST_DATA_FILENAME


# In[39]:


with open(label_encoder_filename, 'rb') as file:  
    le = pickle.load(file)


# In[40]:


le.inverse_transform(1)


# In[41]:


# Load model from file
with open(model_filename, 'rb') as file:  
    model = pickle.load(file)


# In[42]:


n_gram_list = load_n_grams(filename = 'n_grams.txt')


# In[43]:


data_n_gram_test = prepare_n_gram_dataset(dataset = test_data, ns = [1,2,3], n_gram_list = n_gram_list)


# In[44]:


prediction = model.predict(data_n_gram_test)


# In[45]:


prediction_language = list(le.inverse_transform(prediction))


# In[49]:


#print(test_data[0:30])


# In[50]:


print(prediction_language[0:30])


# In[48]:


len(test_data)

