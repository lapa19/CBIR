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
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans,MiniBatchKMeans

class feature(object):
	def __init__(self):
		self.directory = '/home/aparna/6thsem/ir/project/dataset10'
		self.dirs = os.listdir(self.directory)
		self.output = open('descriptord.pkl', 'wb')
		self.vocabfile= open('vocabd.pkl','wb')
		# Initiate SIFT detector
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.dictionarySize = 2000

	def bag_of_words(self):
	#initiate visual bag of words trainer
		#self.BOW = cv2.BOWKMeansTrainer(self.dictionarySize)
		imid = 0
		k=0
		self.X=[]
		self.kpi = []
		self.desi = []
		self.image_names=[]
		imagefile = open('imagesd.pkl','wb')
		for direc in self.dirs:
			print direc
			dir1 = os.path.join(self.directory,direc)
			imgs = os.listdir(dir1)
			for im in imgs:
				self.image_names.append((im,dir1))
				self.kpi.append([])
				self.desi.append([])
				#print im
				pickle.dump((im,dir1),imagefile,pickle.HIGHEST_PROTOCOL)
				img = skimage.io.imread(os.path.join(dir1,im))
				# find the keypoints and descriptors with SIFT
				kp1, des1 = self.sift.detectAndCompute(img,None)
				self.kpi[-1].append(kp1)
				self.desi[-1].append(des1)
				#print len(des1)
				for d in des1:
					#write each descriptor image name and image id to file
					obj = (d, im, dir1, imid)
					k = k + 1
					pickle.dump(obj, self.output, pickle.HIGHEST_PROTOCOL)
					self.X.append(d)
				# next image
				imid = imid + 1
				#self.BOW.add(des1)

		self.output.close()
		imagefile.close()
		kmeans = MiniBatchKMeans(init='k-means++',n_clusters=self.dictionarySize,batch_size=500,random_state=0).fit(self.X)
		self.voc = kmeans.cluster_centers_
		print "clustered"
		pickle.dump(self.voc,self.vocabfile,pickle.HIGHEST_PROTOCOL)
		self.vocabfile.close()
		#self.voc = self.BOW.cluster()

	
	def encode(self):
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary
		flann = cv2.FlannBasedMatcher(index_params,search_params)
		self.assign=[]
		for i in range(len(self.desi)):
			#self.assign.append([])
			matches = flann.knnMatch(np.asarray(self.desi[i][0],np.float32),np.asarray(self.voc,np.float32),k=1)
			#for t in range(len(matches)):
			self.assign.append([matches[t][0].trainIdx for t in range(len(matches))])

	def pooling(self):
		print "pooling"
		self.histog=[]
		doc_word=[]
		docwordfile=open('doc_wordd.pkl','wb')
		histogramfile = open('histod.pkl','wb')
		for i in range(len(self.assign)): 
			hist=[0]*(self.dictionarySize)
			for idx in range(0,self.dictionarySize):
				l = len((np.where(np.array(self.assign[i]) == idx))[0])
				hist[idx] = l
			doc_word.append(hist)
			#hist2 = np.array(hist)/float(sum(hist))
			pickle.dump((self.image_names[i][0],self.image_names[i][1],hist),histogramfile,pickle.HIGHEST_PROTOCOL)
		pickle.dump(doc_word,docwordfile,pickle.HIGHEST_PROTOCOL)
		docwordfile.close()
		histogramfile.close()
		print "pooled"


		


ftObject = feature()
ftObject.bag_of_words()
#ftObject.codeBook()
#ftObject.histogram()
ftObject.encode()
ftObject.pooling()