import pickle
import numpy as np
import gensim
import editdistance
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix
from scipy import io, sparse
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:

    def __init__(self, **kwargs):
        self._classifier = KNeighborsClassifier(n_jobs=6, **kwargs)
        self._official_labels = None

    def fit(self, official_embeddings, official_labels):
        self._classifier.fit(official_embeddings, official_labels)
        self._official_labels = official_labels

    def edit_distance(self, test_string, official_string):
        dist = 0
        word_length_threshold = 3
        for test_word in test_string.split():
            if len(test_word) > word_length_threshold:
                dist += np.min([editdistance.eval(test_word, official_word) for official_word in official_string.split()])
        return dist

    def predict_with_edit_dist(self, test_embeddings, test_descriptions, official_descriptions):
        NN_dists, NN_indices = self._classifier.kneighbors(test_embeddings, return_distance=True)
        ntest_docs = test_embeddings.shape[0]
        nNN = NN_indices.shape[1]
        best_neighbors = np.empty(ntest_docs, dtype=int)
        for i in xrange(ntest_docs):
            test_string = test_descriptions[i]
            edit_dists = [self.edit_distance(test_string, official_descriptions[NN_indices[i,j]]) for j in xrange(nNN)]
            best_neighbors[i] = NN_indices[i,np.argsort(edit_dists)[0]]

        predicted_codes = np.array([self._official_labels[best_neighbors[i]] for i in xrange(ntest_docs)])
        return predicted_codes

    def predict(self, test_embeddings):
        predicted_codes = self._classifier.predict(test_embeddings)
        return predicted_codes
        
        
        
