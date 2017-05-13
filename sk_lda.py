from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import LinearSVC
import os
import joblib
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import skimage.io
import cv2
import matplotlib.pyplot as plt
import datetime

from scipy.spatial.distance import euclidean

np.set_printoptions(threshold=np.nan)

class lda(object):
	def __init__(self):
		self.doc_word = joblib.load("doc_wordd.pkl")
		self.dtdistr = open('dtdistrd.pkl','wb') 


	def lda_sk(self):
		s1 = datetime.datetime.now()
		self.ld = LatentDirichletAllocation(n_topics=20,learning_method='online',max_iter=100)
		self.ld.fit(self.doc_word)
		#print ld.components_
		self.doc_topic_distr = self.ld.transform(self.doc_word)
		pickle.dump((self.doc_topic_distr,self.ld),self.dtdistr,pickle.HIGHEST_PROTOCOL)
		self.dtdistr.close()
		s2 = datetime.datetime.now()
		print "Time taken",s2-s1
	


ldaObj = lda()
ldaObj.lda_sk()
