import numpy as np

# Sample user/rating matrix (rows are users, columns are movies)
user_rating_matrix = np.array([
    [5, 5],
    [5, 0],
    [5, 0],
    [5, 0],
    [4, 0]
])

# Function to calculate average ratings for movies
def calculate_average_ratings(matrix):
    num_users, num_movies = matrix.shape
    average_ratings = np.zeros(num_movies)
    for movie in range(num_movies):
        ratings = matrix[:, movie]
        num_ratings = np.count_nonzero(ratings)
        if num_ratings > 0:
            average_ratings[movie] = np.sum(ratings) / num_ratings
    return average_ratings

def calculate_adjusted_average_ratings(matrix, z):
    num_users, num_movies = matrix.shape
    average_ratings = np.zeros(num_movies)
    for movie in range(num_movies):
        ratings = matrix[:, movie]
        num_ratings = np.count_nonzero(ratings)
        if num_ratings > 0:
            avg_rating = np.sum(ratings) / num_ratings
            std_dev = np.std(ratings)
            adjusted_avg = avg_rating - z * (std_dev / np.sqrt(num_ratings))
            average_ratings[movie] = adjusted_avg
    return average_ratings

# Get the average ratings for movies
average_ratings = calculate_average_ratings(user_rating_matrix)

z = 1.96
adjusted_average_ratings = calculate_adjusted_average_ratings(user_rating_matrix, z)


# Print the average ratings
print("Average ratings for movies:")
for movie_idx, rating in enumerate(average_ratings):
    print(f"Movie {movie_idx + 1}: {rating:.2f}")

# Print the adjusted average ratings
print("Adjusted average ratings for movies:")
for movie_idx, rating in enumerate(adjusted_average_ratings):
    print(f"Movie {movie_idx + 1}: {rating:.2f}")