# MH-Term-Project
## Mental Health Classification with Social Media
## Team-MH
### Members 
* Lei Xian 
* Jiahao Xu
* Yang Shi
## Technology 
* Python
* PyTorch
* Sklearn
* Numpy
* Pandas
* GPU computing
## Problem Statement
Analyse posts from the social media platform Reddit and develope different classifiers to classify posts related to mental illness according to 3 disorder themes and 1 normal class. 
## Data
* Collected posts (text) from Reddit: https://www.reddit.com
* Web scraping (reddit API), manually labeled (150 samples each illness class)
* Classes: 
  * Normal (300)
  * Depression (100)
  * Bipolar disorder (100)
  * PTSD (100)

## Data Cleaning
* Convert to lowercase 
* Lemmatization
* Noise Removal
  * Remove punctuations 
  * Remove links 
  * Remove stop words
  
## Features
* Word count
* TF-IDF on word level
* TF-IDF on n-gram level
* Word Embedding (GloVe with 300 dimension)


## Approach
* Try different models to classify 2 classes
  * normal 
  * illness 

* Try different models to classify 4 classes
  * normal
  * depression 
  * PTSD
  * bipolar disorder  

## Models
* Basic models
  * Naive Bayes
  * SVM witrh different kernels 
  * Random forest
* CNN
* Bidirection LSTM
* LEAM (Label-Embedding Attentive Model)

## Reference
* Wang, Guoyin, et al. "Joint Embedding of Words and Labels for Text Classification." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.
* Schuster, Mike, and Kuldip K. Paliwal. "Bidirectional recurrent neural networks." Signal Processing, IEEE Transactions on 45.11 (1997)
Figure is taken from https://blog.statsbot.co/machine-learning-translation-96f0ed8f19e4
* Ho, Tin Kam (1995). Random Decision Forests (PDF). Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal
Figure is taken from http://www.nrronline.org/viewimage.asp?img=NeuralRegenRes_2018_13_6_962_233433_f2.jpg
* Cortes, Corinna; Vapnik, Vladimir N. (1995). "Support-vector networks". Machine Learning. 20 (3): 273–297. 
Figures are taken from Wikipedia https://en.wikipedia.org/wiki/Support-vector_machine
* Maron, M. E. (1961). "Automatic Indexing: An Experimental Inquiry". Journal of the ACM. 8 (3): 404–417
https://en.wikipedia.org/wiki/Naive_Bayes_classifier
* Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. "Faster r-cnn: Towards real-time object detection with region proposal networks." In Advances in neural information processing systems, pp. 91-99. 2015.

