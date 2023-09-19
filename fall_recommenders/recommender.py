import numpy as np

class Recommender:
    def __init__(self, type, params=None):
        # Initialize the recommender with the chosen type
        self.type = type
        self.params = params
        self.model = None  # You'll store the trained model here

    def set_type(self, type):
        self.type = type

    def set_params(self, params):
        self.params = params

    def add_params(self, params):
        self.params.append(params)

    def check_param_correctness(self):
        # Check if both "pearson-correlation" and "cosine-similarity" are not in params
        # at the same time
        if ["pearson-correlation", "cosine-similarity"] in self.params:
            raise ValueError("Both pearson-correlation and cosine-similarity cannot be used at the same time.")

    def fit(self, training_data):
        # Train the recommender system using your training dataset
        if self.type == "collaborative":
            pass
        elif self.type == "content-based":
            pass
        else:
            self.model = training_data

    def predict(self, user_id, item_id):
        self.check_param_correctness()
        # Generate recommendations for a specific user and item
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.type == "collaborative":
            return Recommender.calculate_nn_colaborative_filtering(self.model, user_id, item_id)
        elif self.type == "content-based":
            # Implement content-based filtering prediction logic
            pass
        elif self.type == "np_collaborative":
            z = 1.96
            return self.calculate_np_collaborative(self.model, z)[item_id]
        elif self.type == "np_collaborative_no_confidence":
            return self.calculate_np_collaborative_no_confidence(self.model)[item_id]
        elif self.type == "lift":
            item_i = user_id
            item_j = item_id
            return self.calculate_lift(self.model, item_i, item_j)
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

    @staticmethod
    def calculate_lift(user_rating_matrix, item_id_i, item_id_j):
        # Calculate lift between two items (item_id_i and item_id_j)
        Ui = np.where(user_rating_matrix[:, item_id_i] > 0)[0]  # Users who rated item i
        Uj = np.where(user_rating_matrix[:, item_id_j] > 0)[0]  # Users who rated item j
        Uij = np.intersect1d(Ui, Uj)  # Users who rated both items i and j
        Uijc = np.setdiff1d(Ui, Uj)  # Users who rated item i but not item j
        if len(Uij) == 0 or len(Uijc) == 0:
            return 1.0  # If either set is empty, lift is 1 (independent)
        lift = (len(Uij) / len(Uj)) / (len(Uijc) / len(Ui))
        return lift
    
    @staticmethod
    def mean_centering_normalization(user_rating_matrix):
        # Normalize the user rating matrix using mean centering
        num_users, num_items = user_rating_matrix.shape
        normalized_matrix = np.zeros((num_users, num_items))
        for user in range(num_users):
            ratings = user_rating_matrix[user, :]
            num_ratings = np.count_nonzero(ratings)
            if num_ratings > 0:
                avg_rating = np.sum(ratings) / num_ratings
                normalized_matrix[user, :] = np.where(ratings != 0, ratings - avg_rating, ratings)
        return normalized_matrix
    
    @staticmethod
    def calculate_cosine_similarity(user_rating_matrix):
        # Calculate the cosine similarity between all pairs of users
        num_users, num_items = user_rating_matrix.shape
        cosine_similarity_matrix = np.zeros((num_users, num_users))
        for user1 in range(num_users):
            for user2 in range(num_users):
                ratings1 = user_rating_matrix[user1, :]
                ratings2 = user_rating_matrix[user2, :]
                cosine_similarity_matrix[user1, user2] = np.dot(ratings1, ratings2) / (np.linalg.norm(ratings1) * np.linalg.norm(ratings2))
        return cosine_similarity_matrix
        

    @staticmethod
    def calculate_nn_colaborative_filtering(user_rating_matrix, user_id, item_id):
        normalized = Recommender.mean_centering_normalization(user_rating_matrix)
        user_similarity_matrix = Recommender.calculate_cosine_similarity(normalized)
        
        num_users, num_items = user_rating_matrix.shape

        s = np.delete(user_similarity_matrix[user_id], user_id)
        r = np.delete(normalized, user_id, axis=0)[:, item_id]

        r_til = np.sum(np.multiply(s, r)) / np.sum(np.abs(s))
        r_mean = np.sum(user_rating_matrix[user_id]) / np.count_nonzero(user_rating_matrix[user_id])
        r_predicted = r_til + r_mean
        return r_predicted

        