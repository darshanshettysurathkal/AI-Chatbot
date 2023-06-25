import random

import nltk
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)


except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  # converts sentences into words
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # we put all the  words in the pattern to words[] and all
    # the patterns to docs and all the tag to labels

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # converting all the words into lower case, then stemming them to convert them into root words
    words = sorted(list(set(words)))
    # set removes duplicate root words

    # In this code snippet, a stemming algorithm (represented by the variable stemmer) is applied to each word in the
    # input list (represented by the variable words). The lower() method is also used to convert all words to
    # lowercase before stemming. This helps to ensure that variations of the same word (e.g., "Jump" and "jump") are
    # treated as the same.

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, docs in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in docs]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Before defining the neural network, tensorflow.compat.v1.reset_default_graph() is called to clear the default
# TensorFlow graph. This is necessary because TensorFlow uses a default graph that accumulates all the operations you
# create in your code. Calling reset_default_graph() ensures that you start with a clean slate, and any previous
# graph is removed from memory. The neural network itself is defined as follows: net = tflearn.input_data(shape=[
# None, len(training[0])]): This line defines the input layer of the neural network with a placeholder that can
# accept any number of data points, where each data point has a number of features equal to len(training[0]). net =
# tflearn.fully_connected(net, 8): This line defines a fully connected layer with 8 neurons and connects it to the
# input layer net. net = tflearn.fully_connected(net, 8): This line defines another fully connected layer with 8
# neurons and connects it to the previous layer net. net = tflearn.fully_connected(net, len(output[0]),
# activation="softmax"): This line defines the output layer with a number of neurons equal to the number of classes
# in the dataset (which is the length of output[0]), and a softmax activation function to output a probability
# distribution over the classes. net = tflearn.regression(net): This line defines the regression layer for training
# the network. Finally, model = tflearn.DNN(net) creates a TensorFlow DNN (Deep Neural Network) model using the
# neural network architecture defined by net. The model object can be used for training, prediction, and evaluation
# on data.


try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat():
    print("start talking with the bot (type quit to quit talking with the bot)")
    while True:
        inp = input("YOU: ")
        if inp.lower() == "quit":
            break
        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = numpy.argmax(result)  # selects the index of the which result has maximum value
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            print("i didnt quite understand, please ask your question with more detail")


chat()
