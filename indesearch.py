import pandas as pd
import numpy as np
import pickle
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__author__ = 'Daniel Villanueva'
__email__ = '2144810@brunel.ac.uk'
__website__ = 'https://www.linkedin.com/in/danielvillanuevanunez/'
__copyright__ = 'Copyright 2022, Daniel Villanueva'

nltk.download('all')

# List of stopwords in Spanish.
stopword_es = nltk.corpus.stopwords.words('spanish')
# Add the "\n" in the list.
stopword_es.append("\n")
# List of punctuations such as ".", ",". "\".
punctuation = [punct for punct in string.punctuation]
# Combine both lists into only one.
undesirable_values = punctuation + stopword_es
# Instantiating the Stemmer class to stem Spanish Words.
stemmer = SnowballStemmer('spanish')

class INDESearch():
           
    def __init__(self, query, top_values, database):
        """ Instantiate the class.
        Args:
            * query(string): text that is input in the search engine.
            * top_values(int): the number of results that you want the search engine to output.
            * df_corpus(dataframe): dataframe containing the vectorized documents.
        """
        self.query = query
        self.top_values = top_values
        self.database = database.copy()

    def cleaning_query_tfidf(self, tfidf_model_location):
        """ Clean the text that is input in the search engine.
        Returns:
            clean_query(numpy array): text converted into an array using the tfidf transformation.
        """
        # Load tf-idf model
        loaded_tfidf = pickle.load(open(tfidf_model_location, 'rb'))
        # 1. Removing strings from the query that belong to the undersirable_values list.
        # 2. Stemming each word.
        query = [stemmer.stem(word) for word in word_tokenize(self.query) if word not in undesirable_values]
        # Unifying the query.
        query = " ".join(query)
        # Conver query into tf.idf
        clean_query = loaded_tfidf.transform([query]).toarray()
        
        return clean_query
    
    def cleaning_query_doc2vec(self, doc2vec_model_location):
        """ Clean the text that is input in the search engine.
        Returns:
            clean_query(numpy array): text converted into an array using the doc2vec transformation.
        """
        # Load tf-idf model
        loaded_doc2vec = Word2Vec.load(doc2vec_model_location)
        # 
        doc = [word.lower() for word in word_tokenize(self.query) if word not in undesirable_values]
        # Conver query into tf.idf
        clean_query = loaded_doc2vec.infer_vector(doc).reshape(-1,1)
        
        return clean_query
    
    def similarity(self, clean_query):
        """ This function calculates the similarity score between the query and the corpus of documents and returns 
            the top top_values most similar documents.
        Args:
            * clean_query(string): cleaned query
        Returns:
            * result(dataframe): top n most similar documents.
        """
        # Convert the database into an array.
        array_database = np.array(self.database.iloc[:,:-1])
        # Calculate similarity score between each document and the query.
        self.database["similarity_score"] = cosine_similarity(array_database, clean_query.reshape(1,-1))[:,0]
        # Sort values by their similarity score.                                               
        score_df = self.database.sort_values(by=["similarity_score"], ascending=False)
        # Output the top values.
        result = score_df.iloc[:self.top_values, -2:].reset_index(drop=True)
        
        return result
