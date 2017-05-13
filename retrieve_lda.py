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
from scipy.spatial.distance import euclidean,cosine

np.set_printoptions(threshold=np.nan)

class lda(object):
	def __init__(self):
		self.directory = '/home/aparna/6thsem/ir/project/dataset10'
		self.doc_topic_distr,self.ld = joblib.load('dtdistrd.pkl') 
		f = open('imagesd.pkl', 'rb')
		self.voc = joblib.load("vocabd.pkl")
		self.dictionarySize=2000
		self.images=[]
		tuples = pickle.load(f)
		while tuples:
			self.images.append(tuples)
			try:
				tuples = pickle.load(f)
			except:
				f.close()
				break

	def encode(self,desi):
		print "encoding"
		assign=[]
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		matches = flann.knnMatch(np.asarray(desi,np.float32),np.asarray(self.voc,np.float32),k=1)
		#for t in range(len(matches)):
		assign.append([matches[t][0].trainIdx for t in range(len(matches))])
		print "encoded"
		return assign
	

	def pooling(self,assign):
		print "pooling"
		hist=[0]*(self.dictionarySize)
		for idx in range(0,self.dictionarySize):
			l = len((np.where(np.array(assign) == idx))[0])
			hist[idx] = l
		#print hist
		return hist 
	
	def get_query_hist(self,img):
		sift = cv2.xfeatures2d.SIFT_create()
		kp1, des1 = sift.detectAndCompute(img,None)
		assign = self.encode(des1)
		hist = self.pooling(assign)
		#print hist
		dq = self.ld.transform((np.array(hist)).reshape(1,-1))
		print dq
		return dq

	def cos_similarity(self):
		im1 = 'image_0004.jpg'
		#im2 = 'image_0009.jpg'
		self.q_class='camera'
		self.query1 = skimage.io.imread(os.path.join(self.directory+'/'+self.q_class,im1))
		doc_topic_query = self.get_query_hist(self.query1)
		de=[]
		dc=[]		

		#print self.doc_topic_distr[0]
		for i in range(len(self.doc_topic_distr)):
			de.append((self.images[i],euclidean(doc_topic_query[0],self.doc_topic_distr[i])))
			dc.append((self.images[i],cosine(doc_topic_query[0],self.doc_topic_distr[i])))
		self.des = sorted(de, key=lambda x: x[1])[:10]
		self.dcs = sorted(dc, key=lambda x: x[1])[:10]

	def plot(self):
		print np.array(self.des)
		fig = plt.figure()
		fig.add_subplot(5,3,1)
		imgplot = plt.imshow(self.query1)
		for i in range(2,12):
			a=fig.add_subplot(5,3,i)
			img1 = skimage.io.imread(os.path.join(self.des[i-2][0][1],self.des[i-2][0][0]))
			imgplot = plt.imshow(img1)
		plt.show()
		print np.array(self.dcs)
		fig = plt.figure()
		fig.add_subplot(5,3,1)
		imgplot = plt.imshow(self.query1)
		for i in range(2,12):
			a=fig.add_subplot(5,3,i)
			img1 = skimage.io.imread(os.path.join(self.dcs[i-2][0][1],self.dcs[i-2][0][0]))
			imgplot = plt.imshow(img1)
		plt.show()

	def precision(self):
		corre=0
		corrc=0
		for i in range(0,len(self.des)):
			corre += self.des[i][0][1].split('/')[-1] == self.q_class
			corrc += self.dcs[i][0][1].split('/')[-1] == self.q_class
		precisione = float(corre)/10
		print precisione
		
		precisionc = float(corrc)/10
		print precisionc


ldaObj = lda()
ldaObj.cos_similarity()
ldaObj.plot()
ldaObj.precision()