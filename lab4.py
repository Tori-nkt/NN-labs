import os

import keras
from keras.datasets import reuters
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Embedding, Dropout, Activation, LSTM, Bidirectional, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import collections

from matplotlib import pyplot
from tensorflow import optimizers

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, **k)

(x_tot, y_tot), (_, _) = reuters.load_data(
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=150,
                                                         test_split=0,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=2)

# Labels in dataset (y_tot) are indexes. From https://github.com/keras-team/keras/issues/12072, we have read the actual strings
classidx =  {'copper': 6, 'livestock': 28, 'gold': 25, 'money-fx': 19, 'ipi': 30, 'trade': 11, 'cocoa': 0, 'iron-steel': 31, 'reserves': 12, 'tin': 26, 'zinc': 37, 'jobs': 34, 'ship': 13, 'cotton': 14, 'alum': 23, 'strategic-metal': 27, 'lead': 45, 'housing': 7, 'meal-feed': 22, 'gnp': 21, 'sugar': 10, 'rubber': 32, 'dlr': 40, 'veg-oil': 2, 'interest': 20, 'crude': 16, 'coffee': 9, 'wheat': 5, 'carcass': 15, 'lei': 35, 'gas': 41, 'nat-gas': 17, 'oilseed': 24, 'orange': 38, 'heat': 33, 'wpi': 43, 'silver': 42, 'cpi': 18, 'earn': 3, 'bop': 36, 'money-supply': 8, 'hog': 44, 'acq': 4, 'pet-chem': 39, 'grain': 1, 'retail': 29}

# Number of samples and classes
num_classes = max(y_tot) - min(y_tot) + 1
print('# of Samples: {}'.format(len(x_tot)))
print('# of Classes: {}'.format(num_classes))

# Reverse dictionary to see words instead of integers
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”
word_to_wordidx = reuters.get_word_index(path="reuters_word_index.json")
word_to_wordidx = {k:(v+2) for k,v in word_to_wordidx.items()}
word_to_wordidx["<PAD>"] = 0
word_to_wordidx["<START>"] = 1
word_to_wordidx["<UNK>"] = 2
wordidx_to_word = {value:key for key,value in word_to_wordidx.items()}
classidx_to_class = {value:key for key,value in classidx.items()}

# Number of words
print('# of Words (including PAD, START and UNK): {}'.format(len(word_to_wordidx)))


# Now we decode the newswires, using the wordidx_to_word dictionary
def decode_newswire(sample):
    """
    Decodes a Newswire

    Arguments:
    sample -- one of the samples in the reuters dataset

    Returns:
    decode_newswire -- a string representing the newswires
    """
    return ' '.join([wordidx_to_word[wordidx] for wordidx in sample])


decoded_newswires = [decode_newswire(sample) for sample in x_tot]

# We print some examples to check if everything is ok
example_num = 1234
print ("ENCODED: ", x_tot[example_num])
print("\nNEWSWIRE: ", decoded_newswires[example_num])
print ("\nCLASS: ", classidx_to_class[y_tot[example_num]])


# We do some statistcs of the length of the documents

documents_word_lenght = [len (sample) for sample in x_tot]
documents_ch_lenght = [len (sample) for sample in decoded_newswires]

f, (ax1, ax2) = plt.subplots(1, 2)
f.suptitle('Document Length Distribution')
f.set_size_inches((20, 5))
ax1.hist(documents_word_lenght, bins=100)
ax1.set_title('In words')
ax2.hist(documents_ch_lenght, bins=100)
ax2.set_title('In characters')

plt.show()

print('Mean Lenght (in words): {}'.format(np.mean(documents_word_lenght)))
print('Mean Lenght (in characters): {}'.format(np.mean(documents_ch_lenght)))
print('Max Lenght (in words): {}'.format(np.max(documents_word_lenght)))
print('Max Lenght (in characters): {}'.format(np.max(documents_ch_lenght)))
print('Min Lenght (in words): {}'.format(np.min(documents_word_lenght)))
print('Min Lenght (in characters): {}'.format(np.min(documents_ch_lenght)))


# Class Distribution

y_hist, y_bin_edges =  np.histogram(y_tot, bins=num_classes)

sorted_num_of_ocurrences = np.sort(y_hist)
sorted_classes = [classidx_to_class[key] for key in np.argsort(y_hist)]

plt.figure(num=None, figsize=(10, 10), dpi=80)
plt.barh(sorted_classes, sorted_num_of_ocurrences, align='center')
plt.yticks(np.arange(num_classes), sorted_classes)
plt.xlabel('Number of Ocurrences')
plt.title('Class Distribution')

ax = plt.gca()
for i, v in enumerate(sorted_num_of_ocurrences):
    ax.text(v + 3, i-0.25, str(v), color='blue')

plt.show()

# Words Distribution

top = 50

words_with_repetition = []
for x in x_tot:
    words_with_repetition.extend(x[1:])

x_hist, x_bin_edges =  np.histogram(words_with_repetition, bins=len(word_to_wordidx), range=(0,len(word_to_wordidx)-1))

sorted_num_of_ocurrences = np.sort(x_hist)[-top:]
sorted_words = [wordidx_to_word[key] for key in np.argsort(x_hist)[-top:]]

plt.figure(num=None, figsize=(10, 10), dpi=80)
plt.barh(sorted_words, sorted_num_of_ocurrences, align='center')
plt.yticks(np.arange(top), sorted_words)
plt.xlabel('Number of Ocurrences')
plt.title('Top {} Words Distribution'.format(top))

ax = plt.gca()
for i, v in enumerate(sorted_num_of_ocurrences):
    ax.text(v + 3, i-0.25, str(v), color='blue')

plt.show()

num_sentences_with_label_in_it = 0


def count_num_sentences_with_label_in_it(x_tot):
    """
    Generator for counting number of sentences which contain the class word within the text

    Arguments:
    sample -- total dataset

    Returns:
    decode_newswire -- generator with True or False
    """
    for x in x_tot:
        words = [wordidx_to_word[wordidx] for wordidx in x]
        for label in classidx:
            yield (label in words)


num_sentences_with_label_in_it = np.sum(count_num_sentences_with_label_in_it(x_tot))

print(
    '{:.2f}% of the examples have the label within the text.'.format(100 * num_sentences_with_label_in_it / len(x_tot)))


top_classes = sorted_classes[-10:]
print("Top 10 most frequent classes: ", top_classes)
print("Dataset will now be filtered by those classes")
top_indexes = [classidx[label] for label in top_classes]
to_keep = [i for i,x in enumerate(y_tot) if x in top_indexes]
y_tot_filtered = y_tot[to_keep]
x_tot_filtered = x_tot[to_keep]
print('# of Samples kept: {}'.format(len(x_tot_filtered)))


print("We will only keep newswires with max 1000 words")
documents_word_lenght = [len (sample) for sample in x_tot_filtered]
to_keep = [idx for idx, value in enumerate(documents_word_lenght) if value <= 1000]
y_tot_filtered = y_tot_filtered[to_keep]
x_tot_filtered = x_tot_filtered[to_keep]
print('# of Samples kept: {}'.format(len(x_tot_filtered)))

from tensorflow.keras.utils import to_categorical
# Tokenizing
num_words_to_tokenize = len(word_to_wordidx)
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=num_words_to_tokenize)
x_tot_matrix = tokenizer.sequences_to_matrix(x_tot_filtered, mode='binary')
y_tot_categorical = to_categorical(y_tot_filtered, num_classes)
print ("When tokenizing the examples, we are just marking which words appear in the sentence. Therefore, we lose the information about the sequence itself, that is, the information which is implied in the order of the words.")


# Test Split
test_split = 0.1
test_num = round(len(x_tot_filtered)*test_split)
x_test_matrix = x_tot_matrix[:test_num]
x_test_seq = x_tot_filtered[:test_num]
x_train_matrix = x_tot_matrix[test_num:]
x_train_seq = x_tot_filtered[test_num:]
y_test_cat = y_tot_categorical[:test_num]
y_train_cat = y_tot_categorical[test_num:]
print ("We keep aside some examples for testing.")


# Padding
documents_word_lenght = [len (sample) for sample in x_tot_filtered]
maxlen = np.max(documents_word_lenght)
x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen)
x_test_pad =  pad_sequences(x_test_seq, maxlen=maxlen)
print ("Padding is for making all sequences of the same length, so we can then use them as inputs of neural networks.")


# We define two functions that will use later

def loadGloveModel(gloveFile):
    """
    Loads GloVe Model

    Arguments:
    gloveFile -- path to the glove file

    Returns:
    model -- a word_to_vec_map, where keys are words, and values are vectors (represented by arrays)
    """

    print("Loading Glove Model")
    f = open(gloveFile, 'r', encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def pretrained_embedding_layer(word_to_vec_map, word_to_wordidx):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_wordidx -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_wordidx) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros(((vocab_len, emb_dim)))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_wordidx.items():
        if word in word_to_vec_map:
            emb_matrix[index, :] = word_to_vec_map[word]
        else:
            emb_matrix[index, :] = word_to_vec_map[
                "random"]  # just to set something when work is not in word_to_vec_map

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

word_to_vec_map = loadGloveModel('glove.6B.50d.txt')

sentence_indices = Input(shape=(maxlen,), dtype='int32')

# Variable-length int sequences.


# Embedding lookup.
vocab_len = len(word_to_wordidx) + 1  # adding 1 to fit Keras embedding (requirement)
emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)
token_embedding = Embedding(input_dim=vocab_len, output_dim=emb_dim)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(sentence_indices)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(sentence_indices)

# CNN layer.
cnn_layer = keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

# Concatenate query and document encodings to produce a DNN input layer.
input_layer = keras.layers.Concatenate()(
    [query_encoding, query_value_attention])
X = Dense(num_classes, activation="linear")(input_layer)
model = Model(inputs = sentence_indices, outputs = X)

#opt = optimizers.Adam(0.01)
#categorical_crossentropy

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 15
validation_split = 0.1

history = model.fit(x_train_pad, y_train_cat, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1,
                    validation_split=validation_split)

score = model.evaluate(x_test_pad, y_test_cat, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot diagnostic learning curves
def summarize_diagnostics(history):
    plt.figure()
    plt.title('MSE Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.show()

    plt.figure()
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()

summarize_diagnostics(history)

model_name = 'weights_attention.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

import json

model_json = model.to_json()
with open("model_attention.json", "w") as json_file:
    json_file.write(model_json)

from keras.models import model_from_json
json_file = open('model_attention.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/weights_attention.h5")
print("Loaded model from disk")

# evaluate loaded model on test data

loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
scores = loaded_model.evaluate(x_test_pad, y_test_cat, batch_size=batch_size, verbose=1)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))
