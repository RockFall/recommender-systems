# Testing the recommender system

from fall_recommenders import Recommender
import numpy as np

# Load the data
user_rating_matrix = np.array([
    [5, 5, 5, 0, 5, 0, 1],
    [5, 0, 5, 0, 4, 0, 0],
    [5, 0, 5, 3, 5, 0, 0],
    [5, 0, 1, 0, 2, 4, 0],
    [5, 0, 1, 0, 2, 0, 5],
    [5, 0, 1, 0, 5, 4, 0],
    [5, 0, 1, 0, 1, 0, 5]
])
# Create a recommender system
np_colab = Recommender(type="np_collaborative")
np_colab_no_confidence = Recommender(type="np_collaborative_no_confidence")

# Train the recommender system
np_colab.fit(user_rating_matrix)
np_colab_no_confidence.fit(user_rating_matrix)

# Predict a rating for user 4 and item 1
print("Predicted rating:")
print("With confidence:")
print(np_colab.predict(user_id=4, item_id=0))
print(np_colab.predict(user_id=4, item_id=1))
print("Without confidence:")
print(np_colab_no_confidence.predict(user_id=4, item_id=0))
print(np_colab_no_confidence.predict(user_id=4, item_id=1))