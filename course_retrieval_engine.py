import pandas as pd
import numpy as np
import pickle
from textwrap import wrap
import re
from typing import Tuple, List
import json

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

# Source for Vader Sentiment Analyzer: https://github.com/cjhutto/vaderSentiment?source=post_page

class CourseRetrievalEngine():
    """
    Uses two document embedding methods Doc2Vec and ti-idf to calculate cosine similarity score between
    the courses in our corpus and the user's input ideal course description. Utilize sentiment analysis
    to separate the negative sentences in user's input (ex.: I don't like too much reading) from the
    positive ones to increase accuracy in recommendation.

    """

    def __init__(self):
        self.dv = Doc2Vec.load('models/doc2vec_model')
        self.tf = pickle.load(open('models/tfidf_model.pkl', 'rb'))
        self.svd = pickle.load(open('models/svd_model.pkl', 'rb'))
        self.svd_feature_matrix = pickle.load(open("models/lsa_embeddings.pkl", 'rb'))
        self.doctovec_feature_matrix = pickle.load(open('models/doctovec_embeddings.pkl', 'rb'))
        self.df = df = self.get_df()
        self.hal = sia()

    def get_df(self, file_path: str = 'courses.json'):
        """
        Get course data from json dump from Django database and turn it into a Pandas dataframe.

        :return: Pandas Dataframe.
        """
        # Load description features
        courses_filtered = []
        with open(file_path) as json_file:
            data = json.load(json_file)
            for course in data:
                course_field = course['fields']
                course_field['description'] = course_field['description'].replace('\n', '')
                courses_filtered.append(course_field)

        courses_df = pd.DataFrame(courses_filtered)
        courses_df = courses_df.rename(columns={'number':'id'})
        return courses_df

    @staticmethod
    def stem_words(text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text

    @staticmethod
    def make_lower_case(text):
        return text.lower()

    @staticmethod
    def remove_stop_words(text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text

    @staticmethod
    def remove_punctuation(text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        text = " ".join(text)
        return text

    def get_description_sentiment(self, description: str) -> Tuple:
        """
        Reads the user's ideal course description and divide it into 2 strings:
        positive and negative strings.

        :return : a tuple of (positive string, negative string)
        """
        sentences = re.split('\.|\but', description)
        sentences = [x for x in sentences if x != ""]
        love_str = ""
        hate_str = ""
        for s in sentences:
            sentiment_scores = self.hal.polarity_scores(s)
            if sentiment_scores['neg'] > 0:
                hate_str = hate_str + s
            else:
                love_str = love_str + s
        return love_str, hate_str

    def clean_description(self, description: str) -> str:
        """
        Clean the description that user inputted like how we cleaned the database course
        descriptions (make all words lower case, remove stop words, remove punctuations,
        and make words into stem words).

        : return: the description that has been cleaned.
        """
        description = self.make_lower_case(description)
        description = self.remove_stop_words(description)
        description = self.remove_punctuation(description)
        description = self.stem_words(description)
        return description

    def get_description_tfidf_embedding_vector(self, description: str) -> List:
        """
        Retrieve the tf-idf vector that represents the user's ideal course description.

        : return: a list of floats representing tf-idf values representative of the user's description.
        """
        description_array = self.tf.transform([description]).toarray()
        description_array = self.svd.transform(description_array)
        description_array = description_array[:,0:25].reshape(1, -1)
        return description_array

    def get_description_doctovec_embedding_vector(self, description: str) -> List:
        """
        Retrieve the Doc2Vec embedding representative of the user's ideal course description.

        : return: a list of floats representing tf-idf values representative of the user's description.
        """

        description_array = self.dv.infer_vector(doc_words=description.split(" "), epochs=200)
        description_array = description_array.reshape(1, -1)
        return description_array

    @staticmethod
    def get_similarity_scores(description_array: List, embeddings) -> List:
        """
        Get the cosine similarity scores between the user's ideal course description and the input embedding.

        : return: a matrix (2D array) representing the cosine similarity between the ideal course
        and each course in our corpus.
        """
        cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=description_array,
                                                           dense_output=True))
        cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        cosine_sim_matrix.columns = ["cosine_similarity"]
        return cosine_sim_matrix

    def get_ensemble_similarity_scores(self, description: str):
        """
        Clean the user's input, get the embedding vectors for the input, and get similarity scores for both
        df-itf and Doc2Vec models. Then average them to find the score for each course in corpus.

        : return: a dataframe of similarity scores (average between df-itf and Doc2Vec embeddings
        for each course in corpus.
        """

        description = self.clean_description(description)
        bow_description_array = self.get_description_tfidf_embedding_vector(description)
        semantic_description_array = self.get_description_doctovec_embedding_vector(description)

        bow_similarity = self.get_similarity_scores(bow_description_array, self.svd_feature_matrix)
        semantic_similarity = self.get_similarity_scores(semantic_description_array, self.doctovec_feature_matrix)

        # The resulting similarity score is averaged of the two scores from the models
        ensemble_similarity = pd.merge(semantic_similarity, bow_similarity, left_index=True, right_index=True)
        ensemble_similarity.columns = ["semantic_similarity", "bow_similarity"]
        ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["semantic_similarity"] + ensemble_similarity["bow_similarity"])/2
        ensemble_similarity.sort_values(by="ensemble_similarity", ascending=False, inplace=True)
        return ensemble_similarity

    def get_dissimilarity_scores(self, description: str):
        """
        Clean the user's input, get the embedding vectors for the input, and get simmilarity score for
        the tf-idf model. Then sort the scores from lowest to highest.

        : return: a dataframe of "dissimilarity" scores from the tf-idf model.
        """
        description = self.clean_description(description)
        bow_description_array = self.get_description_tfidf_embedding_vector(description)
        semantic_description_array = self.get_description_doctovec_embedding_vector(description)

        dissimilarity = self.get_similarity_scores(bow_description_array, self.svd_feature_matrix)
        dissimilarity.columns = ["dissimilarity"]
        dissimilarity.sort_values(by="dissimilarity", ascending=False, inplace=True)
        return dissimilarity

    def query_similar_courses(self, description: str, n: int):
        """
        Get the similarity and disimilarity scores from our corpus against the user's ideal
        course description. Then take out the ones that have high disimilarity scores from the
        result based on the negative strings from the description.

        : param description: user's ideal course description.
        : param n: number of courses to return.
        : return: the courses that are most similar to the user's ideal course description.
        """

        love_str, hate_str = self.get_description_sentiment(description)

        similar_courses = self.get_ensemble_similarity_scores(love_str)
        dissimilar_courses = self.get_dissimilarity_scores(hate_str)
        dissimilar_courses = dissimilar_courses.query('dissimilarity > .3')
        similar_courses= similar_courses.drop(dissimilar_courses.index)

        return similar_courses.head(n)

    def view_recommendations(self, recs):
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15,10))
        ax = axes.ravel()

        for i in range(len(recs)):
            single_title = recs.index.tolist()[i]
            single_course = self.df.query('id==@single_title')
            name = single_course.name.values[0]
            description = single_course.description.values[0]
            title = "Course Title: {} \n Description: {}".format(name, description)

            # perfume_image = single_perfume.image_url.values[0]
            # image = io.imread(perfume_image)
            # ax[i].imshow(image)
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            ax[i].set_title("\n".join(wrap(title, 20)))
            ax[i].axis('off')

        plt.show()
