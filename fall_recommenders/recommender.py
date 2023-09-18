import numpy as np

class Recommender:
    def __init__(self, type):
        # Initialize the recommender with the chosen type
        self.type = type
        self.model = None  # You'll store the trained model here

    def fit(self, training_data):
        # Train the recommender system using your training dataset
        if self.type == "collaborative":
            pass
        elif self.type == "content-based":
            pass
        elif self.type == "np_collaborative":
            self.model = training_data
        elif self.type == "np_collaborative_no_confidence":
            self.model = training_data
        else:
            raise ValueError("Invalid recommender type")

    def predict(self, user_id, item_id):
        # Generate recommendations for a specific user and item
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.type == "collaborative":
            # Implement collaborative filtering prediction logic
            pass
        elif self.type == "content-based":
            # Implement content-based filtering prediction logic
            pass
        elif self.type == "np_collaborative":
            z = 1.96
            return self.calculate_np_collaborative(self.model, z)[item_id]
        elif self.type == "np_collaborative_no_confidence":
            return self.calculate_np_collaborative_no_confidence(self.model)[item_id]
        else:
            raise ValueError("Invalid recommender type")
        
    @staticmethod
    def calculate_np_collaborative_no_confidence(matrix):
        num_users, num_items = matrix.shape
        average_ratings = np.zeros(num_items)
        for item in range(num_items):
            ratings = matrix[:, item]
            num_ratings = np.count_nonzero(ratings)
            if num_ratings > 0:
                average_ratings[item] = np.sum(ratings) / num_ratings
        return average_ratings
    
    @staticmethod
    def calculate_np_collaborative(matrix, z):
        num_users, num_items = matrix.shape
        average_ratings = np.zeros(num_items)
        for item in range(num_items):
            ratings = matrix[:, item]
            num_ratings = np.count_nonzero(ratings)
            if num_ratings > 0:
                avg_rating = np.sum(ratings) / num_ratings
                std_dev = np.std(ratings)
                adjusted_avg = avg_rating - z * (std_dev / np.sqrt(num_ratings))
                average_ratings[item] = adjusted_avg
        return average_ratings
