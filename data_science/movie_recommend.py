## Following the siraj raval tutorial on youtube

## Script which writes a movie recommendation

import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

## fetch dataset

data = fetch_movielens(min_rating=4.0)


# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

## Create a model
## warp = Weighted Approximate-Rank Pairwise
## Uses gradient descent

model = LightFM(loss='warp')

#train the model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):

	## Number of users and movies in the training data
	users, items = data['train'].shape

	## Generate recommendations
	for id in user_ids:

		## Movies they already like
		known_positives = data['item_labels'][data['train'].tocsr()[id].indices]

		## Movies our model predicts they will like
		scores = model.predict(id, np.arange(items))

		## Sort the movies by ranking
		top_items = data['item_labels'][np.argsort(-scores)]

		## Print the results
		print ("User %s" % id)
		print ("        Known positives:")

		for x in known_positives:[:3]
			print ("         %s" % x)

		print ("         Recommended:")

		for x in top_items[:3]:
			print ("           %s" % x)

sample_recommendation(model, data, [3, 25, 450])


