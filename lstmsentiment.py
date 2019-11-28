from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

def remove_pun(element):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return element.translate(translator)

def remove_stopwords(element):
    stop_words = set(stopwords.words('english'))
    element = [word.lower() for word in element.split() if word.lower() not in stop_words]
    return " ".join(element)

def main():
	data = pd.read_csv("sub.csv")
	sentiment = data["rating"] >3
	sentiment = sentiment.astype(int)
	#Prepare model data
	model_data = pd.DataFrame({"MaxTrait":data["MaxTrait"],"text": data["text"], "sentiment":sentiment})
	model_data["text"] = model_data["text"].apply(remove_pun)
	model_data["text"] = model_data["text"].apply(remove_stopwords)
	#normalize personality
	per = np.unique(model_data["MaxTrait"])
	nor = [i+1 for i in range(per.shape[0])]
	nor_map = dict(zip(per, nor))
	model_data["MaxTrait"] = model_data["MaxTrait"].map(nor_map)
	model_data["MaxTrait"] = model_data["MaxTrait"]/5
	#LSTM no personality
	max_len = 200
	tokenizer = Tokenizer(num_words = max_len, split = " ")
	tokenizer.fit_on_texts(model_data["text"].values)
	x = tokenizer.texts_to_sequences(model_data["text"].values)
	x = pad_sequences(x)
	y = pd.get_dummies(model_data["sentiment"]).values
	x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.8, random_state = 13)
	#Build LSTM model
	embed_dim = 150
	lstm_out = 200
	model = Sequential()
	model.add(Embedding(max_len, embed_dim,input_length = x.shape[1]))
	model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	print(model.summary())
	#Train model	
	batch_size = 32
	model.fit(x_train, y_train, epochs = 3, batch_size=batch_size, verbose = 2)
	score,acc = model.evaluate(x_test, y_test, verbose = 2, batch_size = batch_size)
	print("score: %.2f" % (score))
	print("acc: %.2f" % (acc))
	#Initializing measure matrix
	pos_count, neg_count, pos_correct, neg_correct = 0, 0, 0, 0
	for x in range(len(x_test)):   
	    result = model.predict(x_test[x].reshape(1,x_test.shape[1]),batch_size=1,verbose = 2)[0]   
	    if np.argmax(result) == np.argmax(y_test[x]):
	        if np.argmax(y_test[x]) == 0:
	            neg_correct += 1
	        else:
	            pos_correct += 1       
	    if np.argmax(y_test[x]) == 0:
	        neg_count += 1
	    else:
	        pos_count += 1
	pos_acc = pos_correct/pos_count
	neg_acc = neg_correct/neg_count
	g_mean = sqrt(pos_acc*neg_acc)
	#save performance
	accuracy = np.array(acc)
	g_out = np.array(g_mean)
	pos = np.array(pos_acc)
	neg = np.array(neg_acc)
	
	#LSTM by personality
	max_len = 200
	tokenizer = Tokenizer(num_words = max_len, split = " ")
	tokenizer.fit_on_texts(model_data["text"].values)
	x = tokenizer.texts_to_sequences(model_data["text"].values)
	x = pad_sequences(x)
	y = pd.get_dummies(model_data["sentiment"]).values
	x = pd.DataFrame(x)
	x.insert(0, "MaxTrait", model_data["MaxTrait"])
	x = x.to_numpy()
	x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.8, random_state = 13)
	#Build LSTM model
	embed_dim = 150
	lstm_out = 200
	model = Sequential()
	model.add(Embedding(max_len, embed_dim,input_length = x.shape[1]))
	model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	print(model.summary())
	#Train model
	batch_size = 32
	model.fit(x_train, y_train, epochs = 3, batch_size=batch_size, verbose = 2)
	score,acc = model.evaluate(x_test, y_test, verbose = 2, batch_size = batch_size)
	print("score: %.2f" % (score))
	print("acc: %.2f" % (acc))
	
	#Initializing measure matrix
	pos_count, neg_count, pos_correct, neg_correct = 0, 0, 0, 0
	for x in range(len(x_test)):
	    
	    result = model.predict(x_test[x].reshape(1,x_test.shape[1]),batch_size=1,verbose = 2)[0]
	   
	    if np.argmax(result) == np.argmax(y_test[x]):
	        if np.argmax(y_test[x]) == 0:
	            neg_correct += 1
	        else:
	            pos_correct += 1
	       
	    if np.argmax(y_test[x]) == 0:
	        neg_count += 1
	    else:
	        pos_count += 1
	pos_acc = pos_correct/pos_count
	neg_acc = neg_correct/neg_count
	g_mean = sqrt(pos_acc*neg_acc)
	accuracy = np.append(accuracy, np.array(acc))
	g_out = np.append(g_out,np.array(g_mean))
	pos = np.append(pos,np.array(pos_acc))
	neg = np.append(neg, np.array(neg_acc))
	output = pd.DataFrame({"accuracy":accuracy, "g_mean":g_out, "pos":pos, "neg":neg})
	output.to_csv("lstmoutput.csv", index = False)


if __name__ == '__main__':
    main()