import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
import random
#nltk.download()

#Read data from dataset
data = pd.read_csv("sub.csv")

#Examine the data
data.describe()
data.head()

stop_words = set(stopwords.words('english')) 
#np.array(stop_words)

text = pd.Series(data["text"])
text.head(20)

def remove_pun(element):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return element.translate(translator)

def remove_stopwords(element):
    element = [word.lower() for word in element.split() if word.lower() not in stop_words]
    return " ".join(element)

text = text.apply(remove_pun)
text = text.apply(remove_stopwords)

#text.size
#text.head

#Postive 1 negative 0
sent_processed = data["rating"] >3
bool_dict = {True:"pos", False:"neg"}
sent_processed = sent_processed.map(bool_dict)

#Prepare model data
#Run model by group
model_data = pd.DataFrame({"MaxTrait":data["MaxTrait"], "sentiment":sent_processed, "text": text})
#model_data.head()

#Create POS tags
def tokenized_tag(element):
    tokenized = sent_tokenize(element)
    for i in tokenized:
        word_list = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(word_list)
    return tagged

#Extract pos sequence
def extract_pos(element):
    seq = [i[1] for i in element]
    seq = tuple(seq)
    return seq

#Process the data for POS-HMM
model_data["text_pos"] = model_data["text"].apply(tokenized_tag)
model_data["seq"] = model_data["text_pos"].apply(extract_pos)

#model_data.head()

#Initializing HMM parameters
#Get all the unique POS tags
possible_observations =list()
for element in model_data["seq"]:
    possible_observations = possible_observations + np.unique(element).tolist()  

#isinstance(possible_observations,list)

#States
states = ("pos", "neg")
#Possible observations
possible_observations = np.unique(possible_observations).tolist()
#Number of observation sequence
quantities_observations = [1] *model_data.shape[0]
observation_tuple = []
observation_tuple.extend([element for element in model_data["seq"]])

# Input initual parameters as Numpy matrices
start_probability = np.matrix('0.5 0.5')
#Aritifitial transistion probabilities
#Need work
transition_probability = np.matrix('0.6 0.4;  0.3 0.7')
#Aritifitial emission probabilities
#Need work
emission_probability = np.matrix('0 0.04 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.58; 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.06 0 0.03 0.03 0.03 0.03 0.03 0.03 0.34')
