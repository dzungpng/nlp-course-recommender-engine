import pandas as pd
import numpy as np
import pickle
from textwrap import wrap
import re

import matplotlib.pyplot as plt
from skimage import io

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

class CourseRetrievalEngine() {

    def __init__(self):
        self.dv = Doc2Vec.load('models/doc2vec_model')
        self.tf = pickle.load(open('models/tfidf_model.pkl', 'rb'))
        self.svd = pickle.load(open('models/svd_model.pkl', 'rb'))
        self.svd_feature_matrix = pickle.load(open("models/lsa_embeddings.pkl", 'rb'))
        self.doctovec_feature_matrix = pickle.load(open('models/doctovec_embeddings.pkl', 'rb'))

    def get_df(self, file_path: str = 'courses.json'):
        """
        Get course data from json dump from Django database and turn it into a Pandas dataframe.

        :return: Pandas Dataframe.
        """
        
}
