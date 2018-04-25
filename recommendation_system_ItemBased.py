import pandas as pd
import numpy as np
import math
import os
import time

# Form a test_matrix from test5.txt file
# list of tuple(user_id,test_row_index,movie_id)


def build_test_matrix(test_input):
    test_matrix = np.zeros((100,1000))
    with open(test_input) as test:
        user_id_movie_id_list = []
        j = 0
        id_old = None
        for line in test:
            s = line.split(' ')
            id_curr = int(s[0])
            if id_old is None:
                id_old = id_curr
            elif id_old != id_curr:
                id_old = id_curr
                j += 1
            if int(s[2]) == 0:
                test_matrix[j][int(s[1]) - 1] = 0
                user_id_movie_id_list.append((s[0],int(j),int(s[1]) - 1))
            else:
                test_matrix[j][int(s[1]) - 1] = int(s[2])
    test.close()
    return (test_matrix, user_id_movie_id_list)


def compute_adj_cosine_similarity(movie_id1, movie_id2):
    users_who_rated_movie1 = np.where(train_matrix[:, movie_id1] > 0)
    users_who_rated_movie2 = np.where(train_matrix[:, movie_id2] > 0)
    users_rated_both_movies = np.intersect1d(users_who_rated_movie1[0], users_who_rated_movie2[0])
    if len(users_rated_both_movies) == 0:
        return 0.0
    # print("users_rated_both_movies :", users_rated_both_movies)
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    similarity_weight = 0.0

    # Euclidean distance for 1 co-rated movie
    if len(users_rated_both_movies) == 1:
        for user in users_rated_both_movies:
            difference = abs(train_matrix[user][movie_id1] - train_matrix[user][movie_id2])
            similarity_weight = 1.0 / (difference + 1)
        return similarity_weight

    for user in users_rated_both_movies:
        term1 = train_matrix[user][movie_id1] - train_user_avg_rating[user]
        term2 = train_matrix[user][movie_id2] - train_user_avg_rating[user]
        numerator += (term1 * term2)
        denominator1 += (term1 ** 2)
        denominator2 += (term2 ** 2)

    if denominator1 == 0 or denominator2 == 0:
        similarity_weight = 0.0
    else:
        similarity_weight = numerator / ((math.sqrt(denominator1) * math.sqrt(denominator2)))
        if similarity_weight > 1:
            similarity_weight = 1.0
    return similarity_weight

# find the movie ids for which ratings are to be predicted from test_matrix
def predict_rating_adj_cosine(top_k_weight_rating_list, test_user_avg_rating):
    denominator = 0.0
    numerator = 0.0
    for wt_rating in top_k_weight_rating_list:
        denominator += wt_rating[0]
        numerator += (wt_rating[0] * wt_rating[1])
    if denominator != 0:
        # print(top_k_weight_rating_list, numerator, denominator)
        p_rating = int(round(numerator/denominator))
    else:
        p_rating = int(round(test_user_avg_rating))
    return p_rating

def train_user_average_rating(train_matrix):
    # {User:AvgRating}
    train_user_avg_rating = {}
    for i in range(len(train_matrix)):
        user_rated_movie_list = np.where(train_matrix[i] > 0)
        train_user_avg_rating[i] = (np.sum(train_matrix[i, :]) * 1.0)/(len(user_rated_movie_list[0]))
    return train_user_avg_rating

def test_user_average_rating(test_matrix):
    test_user_avg_rating = {}
    for i in range(len(test_matrix)):
        user_rated_movie_list = np.where(test_matrix[i] > 0)
        test_user_avg_rating[i] = (np.sum(test_matrix[i, :]) * 1.0) / (len(user_rated_movie_list[0]))
    return test_user_avg_rating


start_time = time.time()
file_dict = {'test5.txt': ('result5.txt',5), 'test10.txt': ('result10.txt',10), 'test20.txt': ('result20.txt',20)}
dir_path = '/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/'

# Form a train_matrix from train.txt file -- train matrix is a 2d numpy array
train_data = pd.read_csv("/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/train.txt", delimiter='\t', header=None)
train_matrix = train_data.as_matrix()
train_transpose_matrix = train_matrix.transpose()

train_user_avg_rating = train_user_average_rating(train_matrix)
similarity_matrix = np.zeros((1000,1000), float)

for in_file, out_tuple in file_dict.items():
    k = 15
    test_input = os.path.join(dir_path, in_file)
    return_tuple = build_test_matrix(test_input)
    test_matrix = return_tuple[0]
    test_user_avg_rating = test_user_average_rating(test_matrix)
    user_id_movie_id_list = return_tuple[1]
    result_file = os.path.join(dir_path, out_tuple[0])
    with open(result_file, 'w') as result:
        for each_tuple in user_id_movie_id_list:
            weight_list = []
            test_user_id = each_tuple[0]
            test_row_number = each_tuple[1]
            movie_to_be_predicted = each_tuple[2]
            test_user_rated_movies = np.where(test_matrix[test_row_number] > 0)
            for other_movie in test_user_rated_movies[0]:
                if (similarity_matrix[other_movie][movie_to_be_predicted]) == 0:
                    weight = compute_adj_cosine_similarity(other_movie, movie_to_be_predicted)
                    similarity_matrix[other_movie][movie_to_be_predicted] = weight
                similarity_weight = similarity_matrix[other_movie][movie_to_be_predicted]
                if similarity_weight != 0 and similarity_weight > 0.7:
                    weight_list.append((similarity_matrix[other_movie][movie_to_be_predicted],test_matrix[test_row_number][other_movie]))
            weight_list.sort(key=lambda x: x[0], reverse=True)
            top_k_weight_list = weight_list[:k]
            if len(weight_list) != 0:
                p_rating = predict_rating_adj_cosine(weight_list, test_user_avg_rating[test_row_number])
            else:
                p_rating = int(round(((np.sum(test_matrix[each_tuple[1]]) * 1.0) / out_tuple[1])))

            if p_rating > 5:
                p_rating = 5
            elif p_rating <= 0:
                p_rating = 1
            result_string = each_tuple[0] +' '+ str(each_tuple[2] + 1) +' '+ str(p_rating)
            print("item based :", result_string)
            result.write(result_string + "\n")
        result.close()

print("Total Running Time : %s seconds " % (time.time() - start_time))