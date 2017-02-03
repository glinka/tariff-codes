
import urllib
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer, HashingVectorizer
from scipy.stats import ks_2samp
from scipy import io, sparse
from matplotlib import colors, colorbar, cm, pyplot as plt, gridspec as gs, tri
import json
import pickle
import matplotlib
matplotlib.style.use('ggplot')
from util_fns import progress_bar
import codecs
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import csv
import re
from time import time
from itertools import compress
# from embedders import HashingEmbedder
from models import HashingEmbedder, word2vecEmbedder

def parse_codes_descriptions():
    """Parses codes and corresponding words from tariff code text doc"""
    tariff_codes_filebase = './data/tariff-codes-2016'

    tariff_code_regex = re.compile('^[0-9]{10}')
    word_description_regex = re.compile('[a-zA-Z]+')

    code_dict = {}
    unit_description_index = 224 # column at which unit description starts (e.g. NO, KG, X...)
    with open(tariff_codes_filebase + '.txt') as file:
        for line in file:
            line = line[:unit_description_index]
            tariff_code =  int(tariff_code_regex.match(line).group(0))
            word_descriptions =  re.findall(word_description_regex, line)
            code_dict[tariff_code] = list(set(word_descriptions))

    pickle.dump(code_dict, open(tariff_codes_filebase + '.pkl', 'w'))
    

def get_data_to_match(id):
    data_codes = None
    data_descriptions = None
    if id is 'slim':
        data_codes = pickle.load(open('./data/data-codes-labeled-small.pkl', 'r'))
        data_descriptions = pickle.load(open('./data/data-descriptions-labeled-small.pkl', 'r'))

    elif id is 'medium':
        data_codes = pickle.load(open('./data/data-codes-labeled.pkl', 'r'))
        data_descriptions = pickle.load(open('./data/data-descriptions-labeled.pkl', 'r'))

    return [data_codes, data_descriptions]

def get_official_data():
    """load official code data"""

    official_info = pickle.load(open('./data/tariff-codes-2016.pkl', 'r'))
    official_codes = np.array(official_info.keys())
    official_descriptions = [' '.join(words) for words in official_info.values()]

    return [official_codes, official_descriptions]


# TODO: use tocoo() instead of find()

def find_max_k_columns(index_scores, k=5):
    """index_scores is a sparse matrix of shape (ndocs, ncodes). this function finds the sum of each row and the index of the maximum"""
    nrows = index_scores.shape[0]
    maxcol_indices = np.zeros((nrows, k), dtype=int)
    for i in range(nrows):
        current_row = sparse.find(index_scores.getrow(i))
        current_row_indices = current_row[1]
        current_row_values = current_row[2]
        nvals_to_sort = current_row_values.shape[0]
            
        if nvals_to_sort > 0:
            if nvals_to_sort > k:
                nvals_to_sort = k

            max_indices = np.argpartition(current_row_values, -nvals_to_sort)[-nvals_to_sort:]
            sorted_max_indices = np.argsort(current_row_values[max_indices])
            maxcol_indices[i,:nvals_to_sort] = current_row_indices[max_indices][sorted_max_indices[::-1]]

    return maxcol_indices


def get_percent_match(matching_matrix, max_k_columns=None, *args):

    if max_k_columns is None:
        max_k_columns = find_max_k_columns(max_k_columns, *args)

    rowsums = sparse.sum(matching_matrix, axis=1)

    return max_k_columns/rowsums


def get_max_k_columns_and_scores(index_scores, k=5):
    nrows = index_scores.shape[0]
    maxcol_indices = np.zeros((nrows, k), dtype=int)
    for i in xrange(nrows):
        current_row = sparse.find(index_scores.getrow(i))
        current_row_indices = current_row[1]
        current_row_values = current_row[2]
        nvals_to_sort = current_row_values.shape[0]
            
        if nvals_to_sort > 0:
            if nvals_to_sort > k:
                nvals_to_sort = k

            max_indices = np.argpartition(current_row_values, -nvals_to_sort)[-nvals_to_sort:]
            sorted_max_indices = np.argsort(current_row_values[max_indices])
            maxcol_indices[i,:nvals_to_sort] = current_row_indices[max_indices][sorted_max_indices[::-1]]

    rowsums = index_scores.sum(axis=1).getA1()

    return [maxcol_indices, rowsums]
    


def match_codes():
    """Matching full dataset in parallel with customizable embedding method"""
    working_directory = './data/'

    data_codes, data_descriptions = get_data_to_match('medium')

    official_codes, official_descriptions = get_official_data()

    # model = HashingEmbedder(analyzer='char', ngram_range=(2,3)) # word2vecEmbedder() 

    models = [HashingEmbedder(), HashingEmbedder(analyzer='char', ngram_range=(2,3)), HashingEmbedder(analyzer='char', ngram_range=(3,4)), HashingEmbedder(analyzer='char', ngram_range=(2,4)), word2vecEmbedder()]
    model_descs = ['Plain hashing embedder', 'Hashing char (2,3)', 'Hashing char (3,4)', 'Hashing char (2,4)', 'word2vec']
    
    for i, model in enumerate(models):
        t1 = time()
        model.embed(official_descriptions, 'official')
        model.embed(data_descriptions, 'data')

        k = 5
        optimally_matching_columns, matching_scores = model.get_max_k_columns_and_scores(k)


        optimally_matching_codes = official_codes[optimally_matching_columns]
        matching_errors = np.abs(data_codes.reshape((-1,1)) - optimally_matching_codes)

        print '------------------------------'
        print model_descs[i], 'took', time() - t1, 'seconds. Performance summary:'
        print 'Percent with first level correct:', np.sum(matching_errors[:,0] < np.power(10, 9))
        print 'Percent with second level correct:', np.sum(matching_errors[:,0] < np.power(10, 7))
    

def parse_csv():
    """Parses huge csv of exported Enigma dataset"""
    data_filebase = './data/ams-summary-2016'

    word_description_regex = re.compile('[a-zA-Z]+')

    description_index = 26 # index of column which contains description text of lading
    code_index = 27 # index of column which contains tariff code (probably missing)

    nrows_per_page = np.power(10, 6)
    lading_tariff_codes = []
    lading_descriptions = []

    powers_of_ten = np.power(10, range(10))

    with open(data_filebase + '.csv') as file:
        print file.readline() # discard header
        # for i in range((current_page-1)*nrows_per_page):
        #     file.readline()
        csv_reader = csv.reader(file)
        for current_page in range(1,20):
            print current_page
            for i in xrange(nrows_per_page):
                progress_bar(i, nrows_per_page)
                line = csv_reader.next()

                # find code, ensure it has ten digits, then add it to list as int (may remove leading zeros)
                # code = line[code_index]
                # if len(code) > 0:
                #     # throw out those few codes that have 
                #     try:
                #         nmissing_digits = 10 - len(code)
                #         code = int(code)*powers_of_ten[nmissing_digits]
                #     except ValueError:
                #         code = 0
                #         continue
                # else:
                #     code = np.nan
                # lading_tariff_codes.append(code)

                # # find all words (defined as consecutive letters in any case) and append to list
                # description = line[description_index]
                # word_descriptions =  ' '.join(re.findall(word_description_regex, description))
                # lading_descriptions.append(word_descriptions)

                lading_tariff_codes.append(line[code_index])
                lading_descriptions.append(line[description_index])

            pickle.dump(lading_tariff_codes, open('./data/data-codes-2016-' + str(current_page) + '.pkl', 'w'))
            pickle.dump(lading_descriptions, open('./data/data-descriptions-2016-' + str(current_page) + '.pkl', 'w'))


def _find_max_columns(index_scores):
    """index_scores is a sparse matrix of shape (ndocs, ncodes). this function finds the sum of each row and the index of the maximum"""
    nrows = index_scores.shape[0]
    rowsums = np.zeros(nrows)
    maxcol_indices = np.zeros(nrows, dtype=int)
    for i in range(nrows):
        current_row = sparse.find(index_scores.getrow(i))
        current_row_indices = current_row[1]
        current_row_values = current_row[2]
        nvals = current_row_values.shape[0]
        rowsum = 0
        maxval = None
        maxcol = 0
        for j in range(nvals):
            rowsum += current_row_values[j]
            if current_row_values[j] > maxval:
                maxval = current_row_values[j]
                maxcol = current_row_indices[j]

        rowsums[i] = rowsum
        maxcol_indices[i] = maxcol

    return [rowsums, maxcol_indices]


def validate_predictions():
    """Loads data generated by analyze_data and assesses accuracy of classification in different hierarchies of tariff codes"""
    working_directory = './data/'
    
    data_codes = np.array(pickle.load(open(working_directory + 'data-codes-2016-1.pkl', 'r')))
    data_codes = data_codes[~np.isnan(data_codes)]

    # # load official code data
    official_info = pickle.load(open(working_directory + 'tariff-codes-2016.pkl', 'r'))
    official_codes = np.array(official_info.keys())
    official_descriptions = [' '.join(words) for words in official_info.values()]

    print 'Loaded all data'

    errors = np.array(())
    current_index = 0
    for i in range(42):
        chosen_cols = np.load(working_directory + 'colindices' + str(i) + '.pkl')
        nentries = chosen_cols.shape[0]
        errors = np.hstack((errors, np.abs(official_codes[chosen_cols] - data_codes[current_index:current_index + nentries])))
        current_index += nentries
        

    # for file in os.listdir(working_directory):
    #     for i 
    #     if file.startswith('colindices') and file.endswith(".pkl"):

    errors.dump('./data/errors.pkl')
            


def analyze_data():
    working_directory = './data/'
    # # load container data

    data_codes = np.array(pickle.load(open(working_directory + 'data-codes-2016-1.pkl', 'r')))
    data_descriptions = [' '.join(words) for words in pickle.load(open(working_directory + 'data-descriptions-2016-1.pkl', 'r'))]

    # only keep labeled points
    kept_indices = ~np.isnan(data_codes)
    data_descriptions = list(compress(data_descriptions, ~np.isnan(data_codes)))
    data_codes = data_codes[kept_indices]

    # # load official code data
    official_info = pickle.load(open('./data/tariff-codes-2016.pkl', 'r'))
    official_codes = np.array(official_info.keys())
    official_descriptions = [' '.join(words) for words in official_info.values()]
    
    # # set up hashing vectorizer and transform official data
    vectorizer = HashingVectorizer(tokenizer=None, preprocessor=None)
    official_descriptions_counts = vectorizer.transform(official_descriptions).transpose().tocsc(copy=False) # sparse (nshipments, nhash) matrix where each row details word counts in a particular shipping entry

    # # loop through container data, transform, and assess best fit based on dot-products with official_descriptions_counts
    nrows_per_chunk = 5000
    nrows = data_codes.shape[0]
    for i, starting_row in enumerate(range(0, nrows, nrows_per_chunk)):

        t1 = time()

        # handle last chunk of data which may not be clean multiple of nrows_per_chunk
        if starting_row + nrows_per_chunk > nrows:
            nrows_per_chunk = nrows - starting_row

        data_descriptions_chunk = data_descriptions[starting_row:starting_row + nrows_per_chunk]

        data_descriptions_chunk_counts = vectorizer.transform(data_descriptions_chunk)
        rowsums, maxcol_indices =_find_max_columns(data_descriptions_chunk_counts*official_descriptions_counts)
        
        rowsums.dump(working_directory + 'rowsums' + str(i) + '.pkl')
        maxcol_indices.dump(working_directory + 'colindices' +str(i) + '.pkl')

        print 'Iteration', i, 'took', (time() - t1)/60, 'seconds'



def analyze_data_temp():
    """TFIDF analysis of codes"""
    # codes = np.array(pickle.load(open('./data/data-codes-2016-1.pkl', 'r')))
    # lading_descriptions = [' '.join(words) for words in pickle.load(open('./data/data-descriptions-2016-1.pkl', 'r'))]
    # lading_descriptions = list(compress(lading_descriptions, ~np.isnan(codes)))
    # pickle.dump(lading_descriptions, open('./data/temp-ld.pkl', 'w'))
    # codes = list(compress(codes, ~np.isnan(codes)))
    # pickle.dump(codes, open('./data/temp-c.pkl', 'w'))

    # print 'saved codes and ld'

    # lading_descriptions = pickle.load(open('./data/temp-ld.pkl', 'r'))
    # codes = pickle.load(open('./data/temp-c.pkl', 'r'))

    # codes_s = codes[:1000]
    # pickle.dump(codes_s, open('./data/temp-cs.pkl', 'w'))
    # lading_descriptions_s = lading_descriptions[:1000]
    # pickle.dump(lading_descriptions_s, open('./data/temp-lds.pkl', 'w'))

    lading_descriptions = pickle.load(open('./data/temp-lds.pkl', 'r'))
    codes = pickle.load(open('./data/temp-cs.pkl', 'r'))

    official_info = pickle.load(open('./data/tariff-codes-2016.pkl', 'r'))
    official_codes = official_info.keys()
    official_descriptions = [' '.join(words) for words in official_info.values()]
    


    vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer()

    t1 = time()
    vectorizer = HashingVectorizer()
    combined_descriptions = list(lading_descriptions)
    combined_descriptions.extend(official_descriptions)
    vectorizer.fit(combined_descriptions)

    tld = vectorizer.transform(lading_descriptions)
    td = vectorizer.transform(official_descriptions).transpose().tocsc(copy=False)

    assigned_indices = tld*td
    print (time() - t1)/60

    io.mmwrite('./data/assigned-data.mtx', assigned_indices)


    rowsums, maxcol_indices = _find_max_columns(assigned_indices)

    ai.dump('./data/ai.pkl')

    ai = np.load('./data/ai.pkl')
    ai[np.isnan(ai)] = 0
    ai = ai.astype(int)
    

    lading_descriptions = pickle.load(open('./data/temp-lds.pkl', 'r'))
    codes = np.array(pickle.load(open('./data/temp-cs.pkl', 'r')))

    official_info = pickle.load(open('./data/tariff-codes-2016.pkl', 'r'))
    official_codes = np.array(official_info.keys())
    official_descriptions = [' '.join(words) for words in official_info.values()]

    code_dict = {oc: od for oc, od in zip(official_codes, official_descriptions)}

    ac = official_codes[maxcol_indices]

    abs_err = np.abs(ac - codes)

    h1 = abs_err < np.power(10, 9)
    h2 = abs_err < np.power(10, 7)
    h3 = abs_err < np.power(10, 5)

    h1ds = [(code_dict[ac[i]], lading_descriptions[i]) for i in range(1000) if h1[i] == True]
    h2ds = [(code_dict[ac[i]], lading_descriptions[i]) for i in h2 if i == True]
    h3ds = [(code_dict[ac[i]], lading_descriptions[i]) for i in h3 if i == True]

    h2ds = [(official_descriptions[i], lading_descriptions[i]) for i in h2 if i == True]
    h3ds = [(official_descriptions[i], lading_descriptions[i]) for i in h3 if i == True]


def dl_data_eda():
    """Do some EDA on tariff data after dl-ing a subset of it"""
    url = 'https://api.enigma.io/v2/data/ytFLazYk3b9MP8IgT6xoFbEqiCqx8S4u6cwDuQ3VJNJqVsNQZRbL5/enigma.trade.ams.summary.2016?page=0'

    # data = []
    # for i in range(1,30000):
    #     progress_bar(i, 30000)
    #     from_index = url.find('page=')
    #     url = url[:from_index+5] + str(i)
    #     data.append(json.load(urllib.urlopen(url))['result'])

    # data = [item for sublist in data for item in sublist]

    # df = pd.DataFrame(data)
    # df.to_pickle('./tariff.pkl')

    df = pd.read_pickle('./tariff.pkl')

    hist = np.empty(df.shape[0])
    for i, description in enumerate(df['description_text'].values):
        hist[i] = len(description.split())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(hist)
    ax.set_title('All data')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(hist[hist < 100], bins=99, normed=True)
    ax.set_title('Descriptions < 100 words')

    df['harmonized_number'] = df['harmonized_number'].astype(float)
    dfl = df[~np.isnan(df['harmonized_number'])]
    print 'Have', dfl.shape[0], 'labeled pts', 1.*dfl.shape[0]/df.shape[0], 'percent of data'
    
    histl = np.empty(dfl.shape[0])
    for i, description in enumerate(dfl['description_text'].values):
        histl[i] = len(description.split())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(histl, bins=range(int(np.max(histl))))
    ax.set_title('Labeled data')
    # plt.hist(histl[hist < 100], bins=99, normed=True)

    plt.show()

if __name__=='__main__':
    # validate_predictions()
    # dl_data_eda()
    # analyze_data()
    # parse_csv()
    match_codes()
