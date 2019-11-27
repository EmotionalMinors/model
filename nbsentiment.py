import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

stop_words = set(stopwords.words('english'))

#Need changing data source
data = pd.read_csv("sub.csv")
data.head()
sentiment = data["rating"] >3
sentiment = sentiment.astype(int)

#Prepare model data
model_data = pd.DataFrame({"MaxTrait":data["MaxTrait"],"text": data["text"], "sentiment":sentiment})
model_data.head()

def remove_pun(element):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return element.translate(translator)

def remove_stopwords(element):
    element = [word.lower() for word in element.split() if word.lower() not in stop_words]
    return " ".join(element)

model_data["text"] = model_data["text"].apply(remove_pun)
model_data["text"] = model_data["text"].apply(remove_stopwords)

#Generate training test split
#0.8 training 0.2 test
#all data
np.random.seed(13)
array = np.random.rand(model_data.shape[0])
train = array > 0.2
test = array <= 0.2
train = model_data[train]
test  = model_data[test]
#Building TF-IDF Naive Bayes Classifier
x_train = train["text"].values
y_train = train["sentiment"].values
x_test = test["text"].values
y_test = test["sentiment"].values
#TF-IDF
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)
print(train_vectors.shape, test_vectors.shape)
#Train basic Naive Bayes Classifier
nb = MultinomialNB().fit(train_vectors, y_train)
#predict values
predicted = nb.predict(test_vectors)
#Use array to store all data
accuracy = np.array(accuracy_score(y_test, predicted))
f1 = np.array(f1_score(y_test, predicted))

for element in np.unique(model_data["MaxTrait"]):
    model_subset = model_data[model_data["MaxTrait"] == element]
    #Generate training test split
    #0.8 training 0.2 test
    #all data
    np.random.seed(19)
    array = np.random.rand(model_subset.shape[0])
    train = array > 0.2
    test = array <= 0.2
    train = model_subset[train]
    test  = model_subset[test]
    #Building TF-IDF Naive Bayes Classifier
    x_train = train["text"].values
    y_train = train["sentiment"].values
    x_test = test["text"].values
    y_test = test["sentiment"].values
    #TF-IDF
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    print(train_vectors.shape, test_vectors.shape)
    #Train basic Naive Bayes Classifier
    nb = MultinomialNB().fit(train_vectors, y_train)
    #predict values
    predicted = nb.predict(test_vectors)
    #Use array to store all data
    accuracy = np.append(accuracy, np.array(accuracy_score(y_test, predicted)))
    f1 = np.append(f1, np.array(f1_score(y_test, predicted)))

#outputing nb result
category = np.append("All", np.unique(model_data["MaxTrait"]))
output = pd.DataFrame({"Category":category, "accuracy":accuracy, "f1":f1})
output.to_csv("nboutput.csv", index = False)
