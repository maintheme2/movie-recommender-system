import collections
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


movie_mapping = pd.read_csv('data/movie_mapping.csv')

print('Please input which dataset you would like to use:')
print('Options: [1] - u1.base, [2] - u2.base, [3] - u3.base, [4] - u4.base, [5] - u5.base')
data_key = int(input())

print('Loading data...')
if data_key == 1:
    train = pd.read_csv('data/u1.base', sep='\t', encoding='ISO-8859-1')
    test = pd.read_csv('data/u1.test', sep='\t', encoding='ISO-8859-1')
elif data_key == 2:
    train = pd.read_csv('data/u2.base', sep='\t', encoding='ISO-8859-1')
    test = pd.read_csv('data/u2.test', sep='\t', encoding='ISO-8859-1')
elif data_key == 3:
    train = pd.read_csv('data/u3.base', sep='\t', encoding='ISO-8859-1')
    test = pd.read_csv('data/u3.test', sep='\t', encoding='ISO-8859-1')
elif data_key == 4:
    train = pd.read_csv('data/u4.base', sep='\t', encoding='ISO-8859-1')
    test = pd.read_csv('data/u4.test', sep='\t', encoding='ISO-8859-1')
elif data_key == 5:
    train = pd.read_csv('data/u5.base', sep='\t', encoding='ISO-8859-1')
    test = pd.read_csv('data/u5.test', sep='\t', encoding='ISO-8859-1')
else:
    print('Invalid input')

print('Data loaded!')

train.columns = ['userID', 'itemID', 'rating', 'timestamp']
test.columns = ['userID', 'itemID', 'rating', 'timestamp']

key_value_dict = movie_mapping.set_index('movie id')['movie title'].to_dict()

train['itemID'] = train['itemID'].map(key_value_dict)
test['itemID'] = test['itemID'].map(key_value_dict)

train = train.groupby(by=['userID', 'itemID'], as_index=False).agg({"rating":"mean"})

counts_train = train['userID'].value_counts()
counts_test = test['userID'].value_counts()

users_to_check = []
for value in counts_train.items():
    if value[1] > 100:
        users_to_check.append(value[0])

for value in counts_test.items():
    if value[0] in users_to_check and value[1] < 100:
        users_to_check.remove(value[0])

movie_ratings = train.pivot(
    index='userID',
     columns='itemID',
      values='rating').fillna(0)

movie_ratings = movie_ratings.rename_axis(None, axis="columns").reset_index()

csr_matr = csr_matrix(movie_ratings.values)

print("Type 'load' to load the existing model, 'train' to train the model:")

inp = input()
if inp == 'load':
    print('Loading model...')
    try:
        knnPickle = open(f'data/models/knn_model_ubase{data_key}.pkl', 'rb')
    except FileNotFoundError:
        print('Model not found or not trained for this data!')
    knn_model = pickle.load(knnPickle)
    knnPickle.close()
    print('Model loaded!')
elif inp == 'train':
    print('Training model...')
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(csr_matr)
    knnPickle = open(f'data/models/knn_model_ubase{data_key}.pkl', 'wb')
    pickle.dump(knn_model, knnPickle)
    knnPickle.close()
    print('Model trained!')

data = pd.read_csv('data/u.data', sep = '\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
data.drop(columns=['timestamp'], inplace=True)

watched = collections.defaultdict(dict)
for i in data.values.tolist():
    watched[i[0]][i[1]] = i[2]

def recommend_movies(user_id, knn, k):
    """
    Recommend movies to a user based on their similarity to other users.

    Args:
        user_id (int): The ID of the user for whom to recommend movies.
        knn (KNN): The KNN model used to find nearest neighbors.
        k (int): The number of nearest neighbors to consider.

    Returns:
        list: A list of tuples, where each tuple contains the predicted rating and the movie ID of a recommended movie.
    """
    # get nearest neigbours of the specified user
    distances, indices = knn.kneighbors(movie_ratings.iloc[user_id-1, :]\
                        .values.reshape(1, -1), n_neighbors = k)
    
    # get films that user has already watched
    user_watched = set(watched[movie_ratings.index[user_id-1]])

    # get movies that were watched by similar users
    neighbours_watched = {}

    for i in range(0, len(distances.flatten())):
        neighbours_watched[movie_ratings.index[indices.flatten()[i]]] = watched[movie_ratings.index[indices.flatten()[i]]].copy()

        for key, v in neighbours_watched[movie_ratings.index[indices.flatten()[i]]].items():
            neighbours_watched[movie_ratings.index[indices.flatten()[i]]][key] = [1 - distances.flatten()[i], v]

    # get movies that were not watched by similar users
    unwatched_films = []
    for u in neighbours_watched:
        a = neighbours_watched[u].keys() - user_watched.intersection(neighbours_watched[u].keys())
        for f in a:
            unwatched_films.append(f)
    
    # Find unwatched films that are common among neighbours
    common_unwatched = [item for item, count in collections.Counter(unwatched_films).items() if count > 1]
    
    # Predict rating the user would give for the unwatched films
    common_unwatched_rating = []
    for f in common_unwatched:
        m = []
        w = []

        for u in neighbours_watched:
            if neighbours_watched[u].get(f) is not None:
                m.append(neighbours_watched[u].get(f)[0]*neighbours_watched[u].get(f)[1])
                w.append(neighbours_watched[u].get(f)[0])

        # calculate predicted rating by taking the weighted average, where the weight is the distance of the neighbour from the user
        common_unwatched_rating.append([np.sum(m)/np.sum(w), f])
    common_unwatched_rating = sorted(common_unwatched_rating, reverse=True)

    return common_unwatched_rating


def calculate_similarity(list1, list2):
    """
    Calculate the similarity between two lists.

    Args:
        list1 (list): The first input list.
        list2 (list): The second input list.

    Returns:
        float: The similarity between the two lists.
    """
    intersection = set(list1) & set(list2)
    similarity = len(intersection) / (len(list1) + len(list2) - len(intersection))
    return similarity


def evaluate_model(knn, test_data, k):
    """
    Evaluate the performance of the KNN model on the movie ratings dataset.

    Args:
        knn (KNN): The KNN model to evaluate.
        test_data (pd.DataFrame): The test data to evaluate the model on.
        k (int): The number of nearest neighbors to consider.

    Returns:
        tuple(float, float, float): The tuple of MAP@K, Mean rating of the recommended films, Mean similarity of the films.
    """

    similarities = []
    predicted_ratings = []
    precisions = []
    for user in users_to_check:
        k = 10
        recommended_movies_with_rating = recommend_movies(user, knn_model, k)
        recommended_movies = [movie_mapping.loc[movie_mapping['movie id'] == f[1], 'movie title'].values[0] for f in recommended_movies_with_rating]
        recommended_movies_ratings = [f[0] for f in recommended_movies_with_rating]

        # get ratings of top k recommended films
        for rate in recommended_movies_ratings[:k]:
            predicted_ratings.append(rate)

        # calculate precision for each user
        relevant_recs_num = 0
        for rate in recommended_movies_ratings:
            if rate >= 4: relevant_recs_num += 1
        precisions.append(relevant_recs_num / len(recommended_movies))
    
        # get films that user has rated but this data was not provided to the model
        test_movies = []
        for value in test.itertuples():
            user_id = value[1]
            movie_title = value[2]
            rating = value[3]
            if user_id == user and rating >= 4:
                test_movies.append((movie_title, rating))
            
        # sort this data by the rating
        test_movies = sorted(test_movies, key=lambda x: x[1], reverse=True)
        # and calculate similarity using set intersection
        similarities.append(calculate_similarity(recommended_movies, [mov[0] for mov in test_movies]))

    return np.mean(precisions), np.mean(predicted_ratings), np.mean([s for s in similarities if s > 0])


print('\n-----')

mapk, mean_rating, mean_similarity = evaluate_model(knn_model, test, 10)

print('Mean average precision @k of the recommended films:\n\t', mapk)
print('\nMean rating of the recommended films:\n\t', mean_rating)
print('\nMean similarity of the films that were not seen by the model, but user has rated them, and recommended films:\n\t', mean_similarity)

print('-----\n')