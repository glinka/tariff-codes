import json
from FlaskFiles import app
from flask import render_template, jsonify, request, g, make_response, send_file
import re
import numpy as np
from scipy import io, sparse
import pickle
from sklearn.feature_extraction.text import HashingVectorizer

import datetime
from StringIO import StringIO
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.dates import DateFormatter

def get_hashing_vectorizer():
    # # set up hashing vectorizer and transform official data
    if not hasattr(g, 'hashing_vectorizer'):
        g.hashing_vectorizer =  HashingVectorizer(tokenizer=None, preprocessor=None)

    return g.hashing_vectorizer

def get_code_info():
    if not hasattr(g, 'code_matrix'):
        # # load official code data
        official_info = pickle.load(open('../data/tariff-codes-2016.pkl', 'r'))
        official_codes = np.array(official_info.keys())
        official_descriptions = [' '.join(words) for words in official_info.values()]

        vectorizer = get_hashing_vectorizer()
        g.code_matrix = vectorizer.transform(official_descriptions).transpose().tocsc(copy=False) # sparse (nshipments, nhash) matrix where each row details word counts in a particular shipping entry
        g.codes = official_codes
        g.descs = official_descriptions

    return [g.codes, g.descs, g.code_matrix]


def _find_max_columns(index_scores, k):
    """index_scores is a sparse matrix of shape (ndocs, ncodes). this function finds the sum of each row and the index of the maximum"""
    nrows = index_scores.shape[0]
    rowsums = np.zeros(nrows)
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
            rowsums[i] = np.sum(current_row_values)

    return [rowsums, maxcol_indices]


@app.route("/simple_png")
def simple_png():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.random.uniform(size=10), np.arange(10), s=500)

    fig_out = StringIO()
    fig.savefig(fig_out)
    fig_out.seek(0)
    return send_file(fig_out, mimetype='image/png')



@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/match_input')
def match_input():
    input = request.args.get('word_desc')
    vectorizer = get_hashing_vectorizer()
    hashed_input = vectorizer.transform([' '.join(re.findall('[a-zA-Z]+', input))])
    codes, descs, matrix = get_code_info()
    k = 3
    fit_info =  _find_max_columns(hashed_input*matrix, k)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(np.random.uniform(size=10), np.arange(10))
    # fig_out = StringIO()
    # fig.savefig(fig_out)
    # fig_out.seek(0)
    # fig_out = send_file(fig_out, mimetype='image/png')

    # print descs[0], codes[0], matrix, hashed_input
    # print ''
    # print sparse.find(hashed_input), hashed_input.shape
    return render_template("output.html", codes=[{'id':codes[i], 'desc':descs[i]} for i in fit_info[1][0]])



if __name__=='__main__':
    app.run()
