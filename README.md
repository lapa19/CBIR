# CBIR
Content based image retrieval using topic modelling

Topic modelling has been used in the past to discover "latent"/hidden topics from a corpus of text documents. In this project, it has been extended to image corpus. Statistical inference process of Latent Dirichlet Allocation has been used for finding topic-word distribution and document-topic distribution of documents. 
When user inputs a query image, the topic distribution is found for this image and compared with topic-distributions of all other documents present in the database using Eucledian distance and Cosine similarity.

1. python 1sift.py = Find SIFT features of all documents. Use Bag of Words approach to create a histogram of words for each document i.e the frequency of each word in each document and store to a file.
2. python sk_lda.py = Find topic distributions of all documents and store them to a file.
3. python retrieval.py = Retrieve similar documents to the input query image using Bag of Words approach
4. python retrieve_lda.py = Retrieve similar documents to the input query image by comparing topic distributions found in step 2.
