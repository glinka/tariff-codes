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

from models import HashingEmbedder, word2vecEmbedder, TfidfEmbedder
from classifiers import KNNClassifier, KNNCombinedClassifier
from utils import coarsen_codes, get_section_codes

nNN = 4
model_hash = TfidfEmbedder() #analyzer='char', ngram_range=(4,5), norm='l2')
model_w2v = word2vecEmbedder()

classifier = KNNCombinedClassifier(n_neighbors=nNN)

official_code_labels = coarsen_codes(model_hash.official_codes)
classifier.fit2(model_w2v.official_embeddings, official_code_labels)

chapter_labels = pickle.load(open('../data/chapter-names.pkl', 'r'))

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/match_input')
def match_input():
    input = request.args.get('word_desc')
    input_str = [' '.join(re.findall('[a-zA-Z]+', input))]
    model_hash.embed_data(input_str)
    if not classifier._fit1:
        classifier.fit1(model_hash.official_embeddings, official_code_labels)
    model_w2v.embed_data(input_str)
    pred_codes, pred_weights, pred_indices = classifier.predict_combined(model_hash.data_embeddings, model_w2v.data_embeddings)
    # keep nonzero codes and ensure everything adds to one after rounding
    mask = pred_weights[0] > 1e-3
    pred_codes = pred_codes[0,mask]
    pred_weights = pred_weights[0,mask]
    pred_weights = np.round(pred_weights, 2)
    pred_weights[-1] = 1 - np.sum(pred_weights[:-1])
    pred_list = []
    for i in xrange(pred_codes.shape[0]):
        pred_list.append({'id':pred_codes[i], 'desc':chapter_labels[pred_codes[i]], 'weight':pred_weights[i]})
    return render_template("output.html", codes=pred_list)


if __name__=='__main__':
    app.run()
