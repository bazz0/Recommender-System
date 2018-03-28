import numpy as np
import pandas as pd
df = pd.read_csv('G:Mckinsey/train.csv', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_challenge = df.challenge_ID.unique().shape[0]
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items))
for i in train_data.iterrows():
    train_data_matrix[i[1]-1, i[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_challenge))
for i in test_data.iterrows():
    test_data_matrix[i[1]-1, i[2]-1] = line[3]
	
from sklearn.metrics.pairwise import pairwise_distances
us = pairwise_distances(train_data_matrix, metric='cosine')
cs = pairwise_distances(train_data_matrix.T, metric='cosine')

def prediction(ratings, similarity, type='user_id'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


cs = predict(train_data_matrix, cs, type='challenge_ID')
us = predict(train_data_matrix, us, type='user_id')


from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))	