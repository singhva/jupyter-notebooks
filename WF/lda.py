import os, sys, re
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import numpy as np
from numpy import argmax
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import time

def main():
	base_dir = os.path.join(os.environ["HOME"], "Desktop", "WF")
	df = pd.read_csv(os.path.join(base_dir, "CFPB-Data.csv"), parse_dates=[0, 13], infer_datetime_format=True)
	df = df[ df["Company"] == "WELLS FARGO BANK, NATIONAL ASSOCIATION" ]
	n_features = None
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=10, max_features=n_features, stop_words='english')
	ignore_pattern = re.compile("[\(\{](.*?)[\)\}]|[X]+")
	print("Scrubbing text......")
	scrubbed_text = [ ignore_pattern.sub("", text) for text in df["Consumer complaint narrative"] ]
	print("Vectorizing....")
	tf = tf_vectorizer.fit_transform(scrubbed_text)
	n_topics = 50
	print("Running LDA with %d topics..." % n_topics)
	start = int(time.time())
	lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=0, n_jobs = -1)
	transformed = lda.fit_transform(tf)
	end = int(time.time())
	print("Done. Time elapsed: %d" % (end - start))
	joblib.dump(lda, os.path.join(base_dir, "pickle", "sklearn_topics-wf-only-50.pkl")) 

if __name__ == "__main__":
	main()