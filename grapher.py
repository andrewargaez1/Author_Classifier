from nltk.corpus import stopwords
import codecs
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import NMF
import pickle
from sklearn import (
    cluster, datasets, 
    decomposition, ensemble, manifold, 
    random_projection, preprocessing)
import matplotlib.pyplot as plt

train  = pd.read_csv('train_df.csv')

def get_data(item):
    data=[]
    for row in item.txt:
        data.append(' '.join(row))
    labels = item.author
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return data, np.array(y)

def vectorizer1(data):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(data).toarray()
    return X, np.array(tfidf.get_feature_names())


data, y = get_data(train)

vect, vocabulary = vectorizer1(data)

print(vocabulary)

ss = preprocessing.StandardScaler()
X_centered = ss.fit_transform(vect)

pca = decomposition.PCA(n_components=10)
X_pca = pca.fit_transform(X_centered)


def plot_mnist_embedding(ax, X, y, title=None):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.
      
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(y[i]), 
                 color=plt.cm.tab10(y[i] / 10.), 
                 fontdict={'weight': 'bold', 'size': 12})

    ax.set_xticks([]), 
    ax.set_yticks([])
    ax.set_ylim([-0.1,1])
    ax.set_xlim([-0.1,.1])

    if title is not None:
        ax.set_title(title, fontsize=16)

pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)


fig, ax = plt.subplots(figsize=(5, 5))
plot_mnist_embedding(ax, X_pca, y)