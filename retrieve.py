import numpy as np
from sklearn.cluster import KMeans
import cv2
import skimage.io
from sklearn.neighbors import DistanceMetric
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
import os
import pickle
import joblib
from scipy.spatial.distance import euclidean,cosine
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class CBIR(object):
	def __init__(self):
		flann_params = dict(algorithm = 1, trees = 5)     
		matcher = cv2.FlannBasedMatcher(flann_params, {})
		self.sift = cv2.xfeatures2d.SIFT_create()		
		#self.bow_extract = cv2.BOWImgDescriptorExtractor(self.sift,matcher)
		self.voc = joblib.load("vocabd.pkl")
		#self.bow_extract.setVocabulary(self.voc)
		self.directory = '/home/aparna/6thsem/ir/project/dataset10'
		self.dictionarySize=2000

	
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
		#hist = np.array(hist)/float(sum(hist))
		print "pooled"
		return hist 


	def query(self):
		self.histofile = open("histod.pkl",'rb')
		im1 = 'image_0004.jpg'
		#im2 = 'image_0009.jpg'
		self.q_class='camera'
		self.query1 = skimage.io.imread(os.path.join(self.directory+'/'+self.q_class,im1))
		kp1, des1 = self.sift.detectAndCompute(self.query1,None)
		a1 = self.encode(des1)
		h1 = self.pooling(a1)
		#print h1

		histo = pickle.load(self.histofile)
		de=[]
		dc=[]
		while histo:
		    #find minimum eucledian distance    
			#dist = (euclidean(histo[2],h1)) #+ euclidean(histo[2],h2))/2.0
			diste = euclidean((np.array(histo[2])),(np.array(h1)))
			distc = cosine((np.array(histo[2])),(np.array(h1)))
			de.append((histo[0],histo[1],diste))
			dc.append((histo[0],histo[1],distc))
			try:
			    histo = pickle.load(self.histofile)
			except:
				self.histofile.close()
				break
		#d = np.array(d)
		
		self.des = sorted(de, key=lambda x: x[2])[:10]
		self.dcs = sorted(dc, key=lambda x: x[2])[:10]
		print np.array(self.des)
		
		fig = plt.figure()
		fig.add_subplot(5,3,1)
		imgplot = plt.imshow(self.query1)
		
		for i in range(2,12):
			a=fig.add_subplot(5,3,i)
			img1 = skimage.io.imread(os.path.join(self.des[i-2][1],self.des[i-2][0]))
			imgplot = plt.imshow(img1)
		plt.show()
		
		print np.array(self.dcs)
		fig = plt.figure()
		fig.add_subplot(5,3,1)
		imgplot = plt.imshow(self.query1)
		
		for i in range(2,12):
			a=fig.add_subplot(5,3,i)
			img1 = skimage.io.imread(os.path.join(self.dcs[i-2][1],self.dcs[i-2][0]))
			imgplot = plt.imshow(img1)
		plt.show()


	def precision(self):
		corre=0
		corrc=0
		for i in range(0,len(self.des)):
			corre += self.des[i][1].split('/')[-1] == self.q_class
			corrc += self.dcs[i][1].split('/')[-1] == self.q_class
		precisione = float(corre)/10
		print precisione
		
		precisionc = float(corrc)/10
		print precisionc

				
cb = CBIR()
cb.query()
cb.precision()