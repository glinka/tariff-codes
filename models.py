from utils import coarsen_codes, get_coarse_code_index_dict
import pickle
import numpy as np
import gensim
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from scipy import sparse
import os

working_directory = './data/'
if 'web' in os.getcwd():
    working_directory = '../data/'
    


class HashingEmbedder():

    def __init__(self, level=1, **kwargs):
        self._level = level
        self._vectorizer = HashingVectorizer(preprocessor=None, non_negative=True, **kwargs)
        self.official_embeddings = None
        self.data_embeddings = None
        self._matching_scores = None

        # load official data
        official_info = pickle.load(open(working_directory + 'tariff-codes-2016.pkl', 'r'))
        self.official_descriptions = [' '.join(words) for words in official_info.values()]
        self.official_codes = np.array(official_info.keys())
        self._ncodes = len(self.official_codes)
        self.official_embeddings = self._embed(self.official_descriptions)


    def _embed(self, text):
        return self._vectorizer.transform(text)

    def embed_data(self, text):
        self.data_embeddings = self._embed(text)
        return self

    def _group_official_codes(self):
        coarse_category_codes = coarsen_codes(self.official_codes)

        official_embeddings_coo = self.official_embeddings.tocoo()

        unique_code_dict = get_coarse_code_index_dict()
        nunique_coarse_category_codes = len(unique_code_dict)

        nword_bins = official_embeddings_coo.shape[1]

        ndata_vals = official_embeddings_coo.data.shape[0]
        rowvals = np.array([unique_code_dict[coarse_category_codes[i]] for i in official_embeddings_coo.row], dtype=int)
        rowcounts = np.zeros(nunique_coarse_category_codes)
        for i in xrange(ndata_vals):
            row = official_embeddings_coo.row[i]
            rowcounts[unique_code_dict[coarse_category_codes[row]]] += 1

        datavals = np.empty(ndata_vals)
        for i in xrange(ndata_vals):
            row = official_embeddings_coo.row[i]
            rowcount = rowcounts[unique_code_dict[coarse_category_codes[row]]]
            datavals[i] = official_embeddings_coo.data[i]/rowcount

        colvals = official_embeddings_coo.col
        return sparse.coo_matrix((datavals, (rowvals, colvals)), shape=(nunique_coarse_category_codes, nword_bins))

class TfidfEmbedder():

    def __init__(self, level=1, **kwargs):
        self._level = level
        self._vectorizer = TfidfVectorizer(preprocessor=None, **kwargs)
        self.official_embeddings = None
        self.data_embeddings = None
        self._matching_scores = None

        # load official data
        official_info = pickle.load(open(working_directory + 'tariff-codes-2016.pkl', 'r'))
        self.official_descriptions = [' '.join(words) for words in official_info.values()]
        self.official_codes = np.array(official_info.keys())
        self._ncodes = len(self.official_codes)

        data_vocab = pickle.load(open(working_directory + 'data-descriptions-labeled.pkl', 'r'))
        self._vectorizer.fit(self.official_descriptions + data_vocab)

        self.official_embeddings = self._embed(self.official_descriptions)


    def _embed(self, text):
        return self._vectorizer.transform(text)

    def embed_data(self, text):
        self.data_embeddings = self._embed(text)
        return self

    def _group_official_codes(self):
        coarse_category_codes = coarsen_codes(self.official_codes)

        official_embeddings_coo = self.official_embeddings.tocoo()

        unique_code_dict = get_coarse_code_index_dict()
        nunique_coarse_category_codes = len(unique_code_dict)

        nword_bins = official_embeddings_coo.shape[1]

        ndata_vals = official_embeddings_coo.data.shape[0]
        rowvals = np.array([unique_code_dict[coarse_category_codes[i]] for i in official_embeddings_coo.row], dtype=int)
        rowcounts = np.zeros(nunique_coarse_category_codes)
        for i in xrange(ndata_vals):
            row = official_embeddings_coo.row[i]
            rowcounts[unique_code_dict[coarse_category_codes[row]]] += 1

        datavals = np.empty(ndata_vals)
        for i in xrange(ndata_vals):
            row = official_embeddings_coo.row[i]
            rowcount = rowcounts[unique_code_dict[coarse_category_codes[row]]]
            datavals[i] = official_embeddings_coo.data[i]/rowcount

        colvals = official_embeddings_coo.col
        return sparse.coo_matrix((datavals, (rowvals, colvals)), shape=(nunique_coarse_category_codes, nword_bins))
        
        
class word2vecEmbedder():

    def __init__(self, level=1, filename='glove.6B.50d.txt'):
        self._level = level
        self.official_embeddings = None
        self.data_embeddings = None
        self._vectorizer = None
        self._ndim = None
        with open(working_directory + filename, "rb") as lines:
            self._vectorizer = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
            self._ndim = self._vectorizer.values()[0].shape[0]

        # load official data
        official_info = pickle.load(open(working_directory + 'tariff-codes-2016.pkl', 'r'))
        self.official_descriptions = [' '.join(words) for words in official_info.values()]
        self.official_codes = np.array(official_info.keys())
        self._ncodes = len(self.official_codes)
        self.official_embeddings = self._embed(self.official_descriptions)


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
        self.data_embeddings = self._embed(text)
        return self

    def _group_official_codes(self, level=1):
        coarse_category_codes = coarsen_codes(self.official_codes)

        unique_code_dict = get_coarse_code_index_dict(level)
        nunique_coarse_category_codes = len(unique_code_dict)

        coarse_category_embeddings = np.zeros((nunique_coarse_category_codes, self._ndim))
        coarse_code_counts = np.zeros(nunique_coarse_category_codes)
        for i in xrange(self._ncodes):
            code_index = unique_code_dict[coarse_category_codes[i]]
            coarse_category_embeddings[code_index] += self.official_embeddings[i]
            coarse_code_counts[code_index] += 1

        coarse_category_embeddings = coarse_category_embeddings.transpose()
        coarse_category_embeddings /= coarse_code_counts
        return coarse_category_embeddings.transpose()
