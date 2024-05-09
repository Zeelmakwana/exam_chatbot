import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Load and preprocess the intents data
intents = json.loads(open('E:\\ddect chatbot\\intents.json').read())

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Preprocess the intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and filter out ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save preprocessed data to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
outputEmpty = [0] * len(classes)

# Create training set with bag of words and one-hot encoding
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=300, batch_size=8, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)
print('Model training completed.')
