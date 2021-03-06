import pickle
import numpy as np
import gensim
import editdistance
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix
from scipy import io, sparse
from sklearn.neighbors import KNeighborsClassifier, LSHForest
from util_fns import progress_bar

class KNNClassifier:

    def __init__(self, **kwargs):
        self._classifier = KNeighborsClassifier(n_jobs=1, **kwargs) # LSHForest(**kwargs)
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

    def predict_with_edit_dist(self, test_embeddings, test_descriptions, official_descriptions, pbar=False):
        chunk_size = 2000
        ntest_docs = test_embeddings.shape[0]
        best_neighbors = np.empty(ntest_docs, dtype=int)
        for start_index in xrange(0, ntest_docs, chunk_size):
            if pbar: progress_bar(start_index, ntest_docs)
            stop_index = start_index + chunk_size
            if stop_index > ntest_docs:
                stop_index = ntest_docs
            
            NN_dists, NN_indices = self._classifier.kneighbors(test_embeddings[start_index:stop_index], return_distance=True)
            nNN = NN_indices.shape[1]
            for i1, i2 in enumerate(xrange(start_index, stop_index)):
                test_string = test_descriptions[i2]
                edit_dists = [self.edit_distance(test_string, official_descriptions[NN_indices[i1,j]]) for j in xrange(nNN)]
                best_neighbors[i2] = NN_indices[i1,np.argsort(edit_dists)[0]]

        predicted_codes = np.array([self._official_labels[best_neighbors[i]] for i in xrange(ntest_docs)])
        return predicted_codes

    def predict(self, test_embeddings, pbar=False):
        chunk_size = 2000
        ntest_docs = test_embeddings.shape[0]
        predicted_codes = np.empty(ntest_docs, dtype=int)
        for start_index in xrange(0, ntest_docs, chunk_size):
            if pbar: progress_bar(start_index, ntest_docs)
            stop_index = start_index + chunk_size
            if stop_index > ntest_docs:
                stop_index = ntest_docs
            predicted_codes[start_index:stop_index] = self._classifier.predict(test_embeddings[start_index:stop_index])
        return predicted_codes

class KNNCombinedClassifier:

    def __init__(self, **kwargs):
        self._classifier1 = KNeighborsClassifier(n_jobs=1, **kwargs)
        self._classifier2 = KNeighborsClassifier(n_jobs=1, **kwargs)
        self._nNN = self._classifier1.get_params()['n_neighbors']
        self._official_labels = None
        self._fit1 = False
        self._fit2 = False

    def fit1(self, official_embeddings, official_labels):
        self._classifier1.fit(official_embeddings, official_labels)
        self._official_labels = official_labels
        self._fit1 = True

    def fit2(self, official_embeddings, official_labels):
        self._classifier2.fit(official_embeddings, official_labels)
        self._official_labels = official_labels
        self._fit2 = True


    def get_recurring_indices(self, ind1, ind2):
        recurring_inds = []
        ninds = ind1.shape[0]
        for i in xrange(ninds):
            combined_indices = np.hstack((ind1[i], ind2[i]))
            unique_indices = np.unique(combined_indices)
            nunique_indices = unique_indices.shape[0]
            recurring_inds.append([unique_indices[i] for i in xrange(nunique_indices) if np.sum(combined_indices == unique_indices[i]) > 1])
        return recurring_inds

    def predict_combined(self, test_embeddings1, test_embeddings2, alpha=0.2, pbar=False):
        chunk_size = 2000
        ntest_docs = test_embeddings1.shape[0]

        predicted_codes = np.empty((ntest_docs,self._nNN) , dtype=int)
        prediction_weights = np.empty((ntest_docs,self._nNN))
        potential_full_codes = []
        for start_index in xrange(0, ntest_docs, chunk_size):
            if pbar: progress_bar(start_index, ntest_docs)
            stop_index = start_index + chunk_size
            if stop_index > ntest_docs:
                stop_index = ntest_docs
            
            NN_dists1, NN_indices1 = self._classifier1.kneighbors(test_embeddings1[start_index:stop_index], return_distance=True)
            NN_dists2, NN_indices2 = self._classifier2.kneighbors(test_embeddings2[start_index:stop_index], return_distance=True)
            probs1, class1 = self.get_assignment_probs(NN_dists1, NN_indices1)
            probs2, class2 = self.get_assignment_probs(NN_dists2, NN_indices2)
            predicted_codes[start_index:stop_index], prediction_weights[start_index:stop_index] = self.predict_from_weights(probs1, probs2, class1, class2, alpha)

            potential_full_codes += self.get_recurring_indices(NN_indices1, NN_indices2)

        return [predicted_codes, prediction_weights, potential_full_codes]

    def get_assignment_probs(self, dists, indices):
        ndocs, nNN = indices.shape
        probs_total = np.zeros((ndocs, nNN))
        classes = np.zeros((ndocs, nNN))
        for i in xrange(ndocs):
            pred_codes = self._official_labels[indices[i]]
            unique_codes = np.unique(pred_codes)
            nunique_pred_codes = unique_codes.shape[0]
            probs = np.zeros(nunique_pred_codes)
            for j in xrange(nunique_pred_codes):
                probs[j] = np.sum(1/dists[i,np.where(pred_codes == unique_codes[j])])
            if np.any(np.isinf(probs)):
                inf_index = np.where(probs == np.inf)
                probs[:] = 0
                probs[inf_index] = 1
            sorted_probs = np.argsort(probs)[::-1]
            stop_index = nNN
            if nunique_pred_codes < nNN:
                stop_index = nunique_pred_codes
            probs_total[i,:stop_index] = probs[sorted_probs[:stop_index]]
            classes[i,:stop_index] = unique_codes[sorted_probs[:stop_index]]

        return probs_total, classes

    def predict_from_weights(self, probs1, probs2, class1, class2, alpha):
        ndocs, nNN = probs1.shape
        predictions = np.zeros((ndocs, nNN),  dtype=int)
        sorted_weights = np.zeros((ndocs, nNN))
        for i in xrange(ndocs):
            unique_classes = np.unique(np.hstack((class1[i], class2[i])))
            nunique_classes = unique_classes.shape[0]
            combined_probs = np.zeros(nunique_classes)
            for j in xrange(nunique_classes):
                combined_probs[j] += np.sum(probs1[i][np.where(class1[i] == unique_classes[j])])
                combined_probs[j] += alpha*np.sum(probs2[i][np.where(class2[i] == unique_classes[j])])
            stop_index = nunique_classes
            if nunique_classes > nNN:
                stop_index = nNN
            sorted_probs = np.argsort(combined_probs)[::-1]
            sorted_weights[i,:stop_index] = combined_probs[sorted_probs[:stop_index]]
            sorted_weights[i] /= np.sum(sorted_weights[i])
            predictions[i,:stop_index] = unique_classes[sorted_probs[:stop_index]]

        return [predictions, sorted_weights]
