# IMDb-Sentiment-Classification
Text sentiment analysis on IMDb movie review data using RNN

### Part-2
Developed LSTM and CNN models to solve a text classification task on movie review data. The goal is to train a classifier that can correctly identify whether a review is positive or negative. The labeled data is located in data/imdb/aclimdb and is split into train (training) and dev (development) sets, which contain 25000 and 6248 samples respectively. For each set, the balance between positive and negative reviews is equal, so that class imbalances do not occur.

In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings. Further, the train and dev sets contain a disjoint set of movies, so no significant performance is obtained by memorizing movie-unique terms and their association with observed labels. In the labeled train/dev sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included.

Glove vectors are stored in the .vector_cache directory.

### Part-3
Developed a recurrent network model to achieve the highest accuracy possible on a holdout test set.
