source of the dataset: http://ai.stanford.edu/~amaas/data/sentiment/
link of GloVe file: http://nlp.stanford.edu/data/glove.6B.zip

sentiment classfication task of movie reviews
This is my final assignment of Python Programming course in 3rd year. 
I tried 2 approaches to complete it parallelly: decision tree and LSTM

In both approaches, data is cleaned and converted using following steps:
1) (data cleaning) remove possible HTML tags, punctuation marks, convert uppercase letters into lowercase letters
2) tokenize text with nltk.word_tokenize
3) remove stop words
4) stemming with nltk.stem.SnowballStemmer

In decision tree approach, I applied feature extraction to the processed data. In details, use sklearn.feature_extraction.text.CountVectorizer to convert a collection of text documents to a matrix of token counts.
Then build the decision tree classifier and train. 75% of dataset used to train, and the rest 25% used to test. The average accuracy on test set was 74%.

In LSTM approach, I use GloVe method to vectorize text and get embedding matrix for each review.
Then, use embedding matrices to train and test LSTM model. Use 11500 of 12500 items as training data, use the rest 1000 items as test data. The average accuracy on test set was about 70%.
