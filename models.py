import numpy as np
import gensim
from sklearn.feature_extraction.text import HashingVectorizer
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


class word2vecEmbedder():

    def __init__(self, filename='glove.6B.200d.txt'):
        self._official_embeddings = None
        self._data_embeddings = None
        self._matching_scores = None
        self._vectorizer = None
        self._ndim = None
        with open('./data/' + filename, "rb") as lines:
            self._vectorizer = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
            self._ndim = self._vectorizer.values()[0].shape[0]


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
        

    def embed(self, text, id):
        if id is 'official':
            self._official_embeddings = self._embed(text).T
        elif id is 'data':
            if self._data_embeddings is None:
                self._data_embeddings = self._embed(text)
            else:
                self._data_embeddings = np.vstack((self._data_embeddings, self._embed(text)))
                self._matching_scores = None
        else:
            print 'ID was neither "official" nor "data". Data will not be embedded.'

        return self



    def get_matching_scores(self):
        if self._matching_scores is None:
            self._matching_scores = np.dot(self._data_embeddings, self._official_embeddings)
        return self._matching_scores

    def get_max_k_columns_and_scores(self, k=5):

        matching_scores = self.get_matching_scores()
        ndocs = matching_scores.shape[0]
        max_indices = np.argpartition(matching_scores, -k, axis=1)[:,-k:]
        sorted_max_indices = np.argsort(matching_scores[np.arange(ndocs).reshape((ndocs,1)), max_indices], axis=1)
        maxcol_indices = max_indices[sorted_max_indices]

        rowsums = np.sum(matching_scores, axis=1)

        return [max_indices, rowsums]
