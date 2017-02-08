import pickle
import numpy as np
import gensim
import editdistance
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix
from scipy import io, sparse

class HashingEmbedder():

    def __init__(self, level=1, **kwargs):
        self._level = level
        self._vectorizer = HashingVectorizer(preprocessor=None, **kwargs)
        self._official_embeddings = None
        self._data_embeddings = None
        self._matching_scores = None

        # load official data
        official_info = pickle.load(open('./data/tariff-codes-2016.pkl', 'r'))
        self._official_descriptions = [' '.join(words) for words in official_info.values()]
        self._official_codes = np.array(official_info.keys())
        self._ncodes = len(self._official_codes)
        self._official_embeddings = self._embed(self._official_descriptions)


    def classify_knn(self, **kwargs):
        classifier = KNeighborsClassifier(n_jobs=4, **kwargs)
        classifier.fit(self._official_embeddings, self.coarsen_codes(self._official_codes))
        pred_codes = classifier.predict(self._data_embeddings)
        return pred_codes

    def edit_distance(self, data_string, official_string):
        dist = 0
        word_length_threshold = 3
        for data_word in data_string.split():
            if len(data_word) > word_length_threshold:
                dist += np.min([editdistance.eval(data_word, official_word) for official_word in official_string.split()])
        return dist

    def edit_dist_knn_classify(self, data_descriptions, **kwargs):
        classifier = KNeighborsClassifier(n_jobs=4, **kwargs)
        coarse_codes = self.coarsen_codes(self._official_codes)
        classifier.fit(self._official_embeddings, coarse_codes)
        nbr_dists, nbr_indices = classifier.kneighbors(self._data_embeddings, return_distance=True)
        ndocs = self._data_embeddings.shape[0]
        nnbrs = nbr_indices.shape[1]
        best_neighbors = np.empty(ndocs, dtype=int)
        for i in xrange(ndocs):
            data_desc = data_descriptions[i]
            edists = np.empty(nnbrs)
            for j in xrange(nnbrs):
                neighbor_desc = self._official_descriptions[nbr_indices[i,j]]
                edists[j] = self.edit_distance(data_desc, neighbor_desc)
            best_neighbors[i] = nbr_indices[i,np.argsort(edists)[0]]
            # print data_descriptions[i]
            # print self._official_descriptions[best_neighbors[i]]
            # print '------------------------------'
        return np.array([coarse_codes[best_neighbors[i]] for i in xrange(ndocs)])
                

    def _embed(self, text):
        return self._vectorizer.transform(text)

    def embed_data(self, text):
        if self._data_embeddings is None:
            self._data_embeddings = self._embed(text)
        else:
            self._data_embeddings = sparse.vstack([self._data_embeddings, self._embed(text)])
            self._matching_scores = None
        return self

    def get_category_matching_scores(self):
        self._matching_scores = self._data_embeddings*self._group_official_codes().transpose(copy=False).tocsc()
        nrows = self._matching_scores.shape[0]
        norms = np.array([np.linalg.norm(self._matching_scores.getrow(i).data) for i in xrange(nrows)])
        norms[norms == 0] = 1
        return sparse.spdiags(1/norms, 0, nrows, nrows)*self._matching_scores

    def get_coarse_code_index_dict(self):
        """Returns dictionary pairing unique, level 'level' codes contained in the official codes with their index in the output embeddings"""
        unique_coarse_category_codes = np.sort(list(set(self.coarsen_codes(self._official_codes))))
        nunique_coarse_category_codes = unique_coarse_category_codes.shape[0]
        unique_code_dict = {code:index for code, index in zip(unique_coarse_category_codes, np.arange(nunique_coarse_category_codes))}
        return unique_code_dict


    def get_coarse_index_code_dict(self):
        return {index:code for code, index in self.get_coarse_code_index_dict().iteritems()}


    def _group_official_codes(self):
        coarse_category_codes = self.coarsen_codes(self._official_codes)

        official_embeddings_coo = self._official_embeddings.tocoo()

        unique_code_dict = self.get_coarse_code_index_dict()
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

    def get_max_k_columns_and_scores(self, k=5):

        matching_scores = self.get_category_matching_scores()

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
                sorted_max_indices = np.argsort(current_row_values[max_indices])[::-1]
                maxcol_indices[i,:nvals_to_sort] = current_row_indices[max_indices][sorted_max_indices]

        rowsums = matching_scores.sum(axis=1).getA1()

        return [maxcol_indices, rowsums]

    def coarsen_codes(self, codes):
        return codes/np.power(10, 2*(5-self._level))


    def get_best_columns(self, epsilon=0.15):
        if self._matching_scores is None:
            self._matching_scores = self.get_category_matching_scores()
        ndocs, ncategories = self._matching_scores.shape
        sorted_best_indices = []
        for i in xrange(ndocs):
            row_coo = self._matching_scores.getrow(i).tocoo()
            kept_vals = row_coo.data > epsilon
            sorted_indices = np.argsort(row_coo.data[kept_vals])[::-1]
            sorted_best_indices.append(row_coo.col[kept_vals][sorted_indices])
        return sorted_best_indices
            
    def get_matching_coarse_codes(self, optimally_matching_columns):
        """best matching cols is list in which ith entry is doc i's cols that exceeded epsilon threshold in 'get_best_columns_and_scores()'"""
        code_dict = self.get_coarse_index_code_dict()
        ndocs = len(optimally_matching_columns)
        matching_codes = [[code_dict[col_index] for col_index in optimally_matching_columns[i]] for i in xrange(ndocs)]
        return matching_codes


    def assess_matching_accuracy(self, true_vals, pred_vals, level=None):
        """pred_vals is list in which ith entry is ith docs' matches that exceeded 'epsilon' threshold in 'get_best_columns_and_scores()', examine minum distance between true_vals and pred_vals"""
        if level == None:
            level = self._level
        elif level > self._level:
            print 'cannot assess accuracy on a lower level than that used for prediction, using default level'
            level = self._level
        elif level < 1:
            print 'min level is one, given level is too low, using default level'
            level = self._level
        ndocs = true_vals.shape[0]
        min_errors = np.array([np.min(np.abs(pred_vals[i] - true_vals[i])/np.power(10, 2*(self._level - level))) if pred_vals[i] else np.nan for i in xrange(ndocs)])
        return min_errors
        
        
    def get_confusion_matrix(self, true_vals, predicted_vals):
        """predicted_vals is shape (ndocs, npreds), true_vals is shape (ndocs,)"""
        categories = set([val/np.power(10,8) for val in true_vals])
        ncategories = len(categories)
        true_vals /= np.power(10,8)
        predicted_vals /= np.power(10,8)

        confusion_matrices = [confusion_matrix(true_vals, predicted_vals[:,i]) for i in xrange(predicted_vals.shape[1])]
        return confusion_matrices


class word2vecEmbedder():

    def __init__(self, level=1, filename='glove.6B.200d.txt'):
        self._level = level
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
        self._official_descriptions = [' '.join(words) for words in official_info.values()]
        self._official_codes = np.array(official_info.keys())
        self._ncodes = len(self._official_codes)
        self._official_embeddings = self._embed(self._official_descriptions)


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
        
    def coarsen_codes(self, codes):
        return codes/np.power(10, 2*(5-self._level))

    def embed_data(self, text):
        if self._data_embeddings is None:
            self._data_embeddings = self._embed(text)
        else:
            self._data_embeddings = np.vstack((self._data_embeddings, self._embed(text)))
            self._matching_scores = None

        return self


    def get_coarse_code_index_dict(self, level=1):
        """Returns dictionary pairing unique, level 'level' codes contained in the official codes with their index in the output embeddings"""
        unique_coarse_category_codes = np.sort(list(set(self.coarsen_codes(self._official_codes))))
        nunique_coarse_category_codes = unique_coarse_category_codes.shape[0]
        unique_code_dict = {code:index for code, index in zip(unique_coarse_category_codes, np.arange(nunique_coarse_category_codes))}
        return unique_code_dict

    def get_coarse_index_code_dict(self, level=1):
        return {index:code for code, index in self.get_coarse_code_index_dict(level).iteritems()}

    def _group_official_codes(self, level=1):
        coarse_category_codes = self.coarsen_codes(self._official_codes)

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

    def get_best_columns(self, epsilon=0.15):
        matching_scores = self.get_category_matching_scores()
        ndocs, ncategories = matching_scores.shape
        sorted_best_indices = []
        indices = np.arange(ncategories)
        for i in xrange(ndocs):
            indices_above_threshold = matching_scores[i] > epsilon
            
            sorted_thresholded_indices = np.argsort(matching_scores[i,indices_above_threshold])[::-1]
            sorted_best_indices.append(indices[indices_above_threshold][sorted_thresholded_indices])
        nmatches = np.array([len(l) for l in sorted_best_indices])
        return sorted_best_indices


    def get_matching_coarse_codes(self, optimally_matching_columns):
        """best matching cols is list in which ith entry is doc i's cols that exceeded epsilon threshold in 'get_best_columns_and_scores()'"""
        code_dict = self.get_coarse_index_code_dict()
        ndocs = len(optimally_matching_columns)
        matching_codes = [[code_dict[col_index] for col_index in optimally_matching_columns[i]] for i in xrange(ndocs)]
        return matching_codes


    def get_max_k_columns_and_scores(self, k=5):

        matching_scores = self.get_category_matching_scores()
        ndocs = matching_scores.shape[0]
        max_indices = np.argpartition(matching_scores, -k, axis=1)[:,-k:]
        row_indices = np.arange(ndocs).reshape((ndocs,1))
        sorted_max_indices = np.argsort(matching_scores[row_indices, max_indices], axis=1)[:,::-1]
        maxcol_indices = max_indices[row_indices, sorted_max_indices]

        rowsums = np.sum(matching_scores, axis=1)

        return [max_indices, rowsums]


    def assess_matching_accuracy(self, true_vals, pred_vals, level=None):
        """pred_vals is list in which ith entry is ith docs' matches that exceeded 'epsilon' threshold in 'get_best_columns_and_scores()', examine minum distance between true_vals and pred_vals"""
        if level == None:
            level = self._level
        elif level > self._level:
            print 'cannot assess accuracy on a lower level than that used for prediction, using default level'
            level = self._level
        elif level < 1:
            print 'min level is one, given level is too low, using default level'
            level = self._level
        ndocs = true_vals.shape[0]
        min_errors = np.array([np.min(np.abs(pred_vals[i] - true_vals[i])/np.power(10, 2*(self._level - level))) if pred_vals[i] != [] else np.nan for i in xrange(ndocs)])
        return min_errors



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

            
    def classify_knn(self, **kwargs):
        classifier = KNeighborsClassifier(n_jobs=4, **kwargs)
        classifier.fit(self._official_embeddings, self.coarsen_codes(self._official_codes))
        pred_codes = classifier.predict(self._data_embeddings)
        return pred_codes


        
        
