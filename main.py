# Testing the recommender system

from fall_recommenders import Recommender
import numpy as np

# Load the data
user_rating_matrix = np.array([
    [5, 5, 5, 0, 5, 2, 1],
    [5, 0, 5, 0, 4, 3, 0],
    [5, 0, 5, 3, 0, 0, 0],
    [5, 0, 1, 0, 0, 0, 0],
    [5, 0, 1, 0, 2, 1, 5],
    [5, 0, 1, 0, 0, 0, 0],
    [5, 0, 1, 0, 1, 0, 5]
])

test = np.array([
    [1.3, -0.7,-0.7, 0],
    [1, 0, 0, -1],
    [0, -0.3, 0.7, -0.3]
])

test2 = np.array([
    [5,3,3,0],
    [5,0,0,3],
    [0,1,2,1]
])

# Create a recommender system
#rec = Recommender(type="collaborative")

# Train the recommender system
#rec.fit(user_rating_matrix)

# Calculate the lift between items 0 and 6, and items 4 and 5
#print(rec.predict(0, 6))
#print(rec.predict(4, 5))

Recommender.calculate_nn_colaborative_filtering(test2, 0, 3)