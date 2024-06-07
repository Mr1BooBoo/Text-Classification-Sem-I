# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:46:46 2023

@author: bilal
"""

import random
import json
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import nltk
#lematization lib
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
f = open(r'C:\Users\bilal\Desktop\Master\NLP\data.json')
intents = json.load(f)

words = []
classes = []
documents = []
ignore = ['?','!',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #get a list of all elements in the dataset and append them to a list along their class
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        #add the classes to a list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#del f, intent, data, pattern
#lematize the data entries
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#bag of words
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


random.shuffle(training)
training = np.array(training)

x_train = list(training[:, 0]) 
y_train = list(training[:, 1])


model = keras.Sequential()
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(len(x_train[0]),1,)))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(len(classes), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#set the optimizer to adam
model.compile(loss='categorical_crossentropy',
              optimizer= keras.optimizers.Adam(learning_rate=0.01),
              metrics=["accuracy"])


history = model.fit(np.array(x_train), np.array(y_train), epochs=15,  batch_size = 5)
model.save('chatbot_model.h5', history)
print('Done')
















































