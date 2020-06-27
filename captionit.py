from numpy import array
from PIL import Image #used to load and create images
import numpy as np
import pickle  #pickle is used to serialize the data
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image #preprocessing of image
from keras.utils import to_categorical
import nltk
import string
from keras.models import Model

model = load_model('D:/caption_bot/model8.h5')
model._make_predict_function()

model_temp = load_model('D:/caption_bot/inception.h5')

model_new = Model(model_temp.input, model_temp.layers[-2].output)
model_new._make_predict_function()

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

max_length=74

# Get training descriptions
doc = open('D:/caption_bot/vocab.txt', 'r').read().strip().split('\n')
vocab = list()

for line in doc:
     vocab.append(line)
     
print('Vocab Length=%d' % len(vocab))
# two functions to get word from index and index fro word
ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            
            preds = model.predict([np.array([image]), np.array(par_caps)])
            
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def greedySearch(photo):
     
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.array([photo]),sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def caption_this_image(image):
    encoded_image = encode(image)
    caption = greedySearch(encoded_image)
    return caption
