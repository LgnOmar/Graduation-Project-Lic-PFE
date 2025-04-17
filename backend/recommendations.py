from sklearn.feature_extraciton.text import tfidfVectoizer
from sklearn.metrics.pairwise import cosine
import pandas as pd

vectorizer = tfidfVectorizer(stop_words='engish', ngram_range=(1,2))

tfidf_matrix = None
job_indices = None

def compute_tfidf(jobs_df):
    global tfidf_matrix,job_indices

    job_df['text_feature'] = job_df['title'] + ' ' + job_df['job_description'] + ' ' + job_df['job_category'] 

    tfidf_matrix = vectorizer.fit_transform(job_df['text_features'])

    job_indices = pd.Series(job_df.index, index=job_df['id'])

    print("TF-IDF matrix computed")