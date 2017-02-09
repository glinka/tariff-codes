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

from models import HashingEmbedder, word2vecEmbedder
from classifiers import KNNClassifier, KNNCombinedClassifier
from utils import coarsen_codes, get_section_codes

# official_info = pickle.load(open('../data/tariff-codes-2016.pkl', 'r'))
# official_codes = np.array(official_info.keys())
# official_descriptions = [' '.join(words) for words in official_info.values()]
nNN = 4
model = HashingEmbedder(analyzer='char_wb', ngram_range=(4,5), norm='l2')
official_code_labels = coarsen_codes(model.official_codes)
classifier = KNNClassifier(n_neighbors=nNN)
classifier.fit(model.official_embeddings, official_code_labels)


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
    model.embed_data([' '.join(re.findall('[a-zA-Z]+', input))])
    pred_code = classifier.predict(model.data_embeddings)
    return render_template("output.html", codes=[{'id':pred_code}])


if __name__=='__main__':
    app.run()
