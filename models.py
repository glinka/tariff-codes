import pickle
import numpy as np
import gensim
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix
from scipy import io, sparse

class HashingEmbedder():

    def __init__(self, **kwargs):
        self._vectorizer = HashingVectorizer(preprocessor=None, **kwargs)
        self._official_embeddings = None
        self._data_embeddings = None
        self._matching_scores = None

    def embed(self, text, id):
        if id is 'official':
            self._official_embeddings = self._vectorizer.transform(text).transpose().tocsc(copy=False)
        elif id is 'data':
            if self._data_embeddings is None:
                self._data_embeddings = self._vectorizer.transform(text)
            else:
                self._data_embeddings = sparse.vstack([self._data_embeddings, self._vectorizer.transform(text)])
                self._matching_scores = None
        else:
            print 'ID was neither "official" nor "data". Data will not be embedded.'

        return self

    def get_matching_scores(self):
        if self._matching_scores is None:
            self._matching_scores = self._data_embeddings*self._official_embeddings

        return self._matching_scores

    def get_max_k_columns_and_scores(self, k=5):

        matching_scores = self.get_matching_scores()

        nrows = self._data_embeddings.shape[0]
        maxcol_indices = np.zeros((nrows, k), dtype=int)
        for i in xrange(nrows):
            current_row = sparse.find(matching_scores.getrow(i))
            current_row_indices = current_row[1]
            current_row_values = current_row[2]
            nvals_to_sort = current_row_values.shape[0]

            if nvals_to_sort > 0:
                if nvals_to_sort > k:
                    nvals_to_sort = k

                max_indices = np.argpartition(current_row_values, -nvals_to_sort)[-nvals_to_sort:]
                sorted_max_indices = np.argsort(current_row_values[max_indices])
                maxcol_indices[i,:nvals_to_sort] = current_row_indices[max_indices][sorted_max_indices[::-1]]

        rowsums = matching_scores.sum(axis=1).getA1()

        return [maxcol_indices, rowsums]



    def get_confusion_matrix(self, true_vals, predicted_vals):
        """predicted_vals is shape (ndocs, npreds), true_vals is shape (ndocs,)"""
        categories = set([val/np.power(10,8) for val in true_vals])
        ncategories = len(categories)
        true_vals /= np.power(10,8)
        predicted_vals /= np.power(10,8)

        confusion_matrices = [confusion_matrix(true_vals, predicted_vals[:,i]) for i in xrange(predicted_vals.shape[1])]
        return confusion_matrices


class word2vecEmbedder():

    def __init__(self, filename='glove.6B.50d.txt'):
        self._official_embeddings = None
        self._data_embeddings = None
        self._matching_scores = None
        self._vectorizer = None
        self._ndim = None
        with open('./data/' + filename, "rb") as lines:
            self._vectorizer = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
            self._ndim = self._vectorizer.values()[0].shape[0]

        # load official data
        official_info = pickle.load(open('./data/tariff-codes-2016.pkl', 'r'))
        official_descriptions = [' '.join(words) for words in official_info.values()]
        self._official_codes = np.array(official_info.keys())
        self._ncodes = len(self._official_codes)
        self._official_embeddings = self._embed(official_descriptions)


    def _embed(self, text):
        ndocs = len(text)
        embeddings = np.zeros((ndocs, self._ndim))
        for i, line in enumerate(text):
            nwords_added = 0
            for word in line.split():
                word_upper = word
                word = word.lower()
                if word in self._vectorizer:
                    embeddings[i] += self._vectorizer[word]
                    nwords_added += 1
                else:
                    other_words = line.split()
                    other_words.remove(word_upper)
                    for other_word in other_words:
                        other_word = other_word.lower()
                        if word + other_word in self._vectorizer:
                            embeddings[i] += self._vectorizer[word + other_word]
                            nwords_added += 1
            embeddings[i] /= nwords_added
        embeddings[np.isnan(embeddings)] = 0
        return embeddings
        
    def embed_data(self, text):
        if self._data_embeddings is None:
            self._data_embeddings = self._embed(text)
        else:
            self._data_embeddings = np.vstack((self._data_embeddings, self._embed(text)))
            self._matching_scores = None

        return self


    def get_coarse_code_index_dict(self, level=1):
        """Returns dictionary pairing unique, level 'level' codes contained in the official codes with their index in the output embeddings"""
        unique_coarse_category_codes = np.sort(list(set(self._official_codes/np.power(10, 2*(5-level)))))
        nunique_coarse_category_codes = unique_coarse_category_codes.shape[0]
        unique_code_dict = {code:index for code, index in zip(unique_coarse_category_codes, np.arange(nunique_coarse_category_codes))}
        return unique_code_dict

    def get_coarse_index_code_dict(self, level=1):
        return {index:code for code, index in self.get_coarse_code_index_dict(level).iteritems()}

    def _group_official_codes(self, level=1):
        coarse_category_codes = self._official_codes/np.power(10, 2*(5-level))

        unique_code_dict = self.get_coarse_code_index_dict(level)
        nunique_coarse_category_codes = len(unique_code_dict)

        coarse_category_embeddings = np.zeros((nunique_coarse_category_codes, self._ndim))
        coarse_code_counts = np.zeros(nunique_coarse_category_codes)
        for i in xrange(self._ncodes):
            code_index = unique_code_dict[coarse_category_codes[i]]
            coarse_category_embeddings[code_index] += self._official_embeddings[i]
            coarse_code_counts[code_index] += 1

        coarse_category_embeddings = coarse_category_embeddings.transpose()
        coarse_category_embeddings /= coarse_code_counts
        return coarse_category_embeddings.transpose()


    def get_category_matching_scores(self, level=1):
        """Scores input documents against official categories up to level 'level', where level 1 indicates the first two digits, level 2 the first four and so on (with max level of five corresponding to the full 10-digit code)."""
        category_embedding = self._group_official_codes(level)
        if self._matching_scores is None:
            self._matching_scores = np.dot(self._data_embeddings, category_embedding.transpose())
            for row in self._matching_scores:
                norm = np.linalg.norm(row)
                if norm > 0:
                    row /= norm
        return self._matching_scores

    def get_max_k_columns_and_scores(self, k=5):

        matching_scores = self.get_category_matching_scores()
        ndocs = matching_scores.shape[0]
        max_indices = np.argpartition(matching_scores, -k, axis=1)[:,-k:]
        row_indices = np.arange(ndocs).reshape((ndocs,1))
        sorted_max_indices = np.argsort(matching_scores[row_indices, max_indices], axis=1)
        maxcol_indices = max_indices[row_indices, sorted_max_indices]

        rowsums = np.sum(matching_scores, axis=1)

        return [max_indices, rowsums]

    def get_best_columns_and_scores(self, epsilon=0.14):
        matching_scores = self.get_category_matching_scores()
        ndocs, ncategories = matching_scores.shape
        sorted_best_indices = []
        indices = np.arange(ncategories)
        for i in xrange(ndocs):
            indices_above_threshold = matching_scores[i] > epsilon
            sorted_thresholded_indices = np.argsort(matching_scores[i,indices_above_threshold])
            sorted_best_indices.append(indices[indices_above_threshold][sorted_thresholded_indices])
        nmatches = np.array([len(l) for l in sorted_best_indices])
        return sorted_best_indices



    def get_confusion_matrix(self, true_vals, predicted_vals):
        """predicted_vals is shape (ndocs, npreds), true_vals is shape (ndocs,)"""
        # true_vals.shape = (true_vals.shape[0],)
        # categories = set([val for val in true_vals])
        # ncategories = len(categories)
        # true_vals = np.copy(true_vals)
        # predicted_vals = np.copy(predicted_vals)
        # true_vals /= np.power(10,8)
        # predicted_vals /= np.power(10,8)

        true_vals.shape = (-1,)
        nvals, npreds = predicted_vals.shape
        final_preds = np.copy(predicted_vals[:,0])
        for i in xrange(nvals):
            deviations = predicted_vals[i] - true_vals[i]
            for j in xrange(npreds):
                if deviations[j] == 0:
                    final_preds[i] = predicted_vals[i,j]
        return confusion_matrix(true_vals, final_preds)

            
        
        
        
