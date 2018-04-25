import pandas as pd
import numpy as np
import math
import os
import time


def iuf_calc(train_matrix):
    iuf_dict = {}
    for i in range(1000):
        rated_users = np.where(train_matrix[:, i] > 0)
        iuf = math.log10(200.0/(len(rated_users[0]) + 1))
        iuf_dict[i] = iuf
    return iuf_dict

# Form a test_matrix from test5.txt file
# list of tuple(user_id,test_row_index,movie_id)


def build_test_matrix(test_input):
    test_matrix = np.zeros((100, 1000), int)
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


# find the movie ids for which ratings are to be predicted from test_matrix
def compute_cosine_similarity_iuf(test_vector,train_vector, iuf_dict):
    comparable_movies = np.where(test_vector > 0)
    test_sum = 0.0
    train_sum = 0.0
    numerator = 0.0
    num_comm_movies = 0
    comm_movie = None
    for movie in comparable_movies[0]:
        if train_vector[movie] > 0:
            num_comm_movies += 1
            comm_movie = movie
            test_sum += ((test_vector[movie] * iuf_dict[movie]) ** 2)
            train_sum += ((train_vector[movie] * iuf_dict[movie]) ** 2)
            numerator += ((train_vector[movie] * iuf_dict[movie]) * (test_vector[movie] * iuf_dict[movie]))

    # Euclidean distance calculation for users with one common movie - bcos cosine sim will be 1 for such cases
    if num_comm_movies == 1:
        euclidean_dist = math.sqrt((test_vector[comm_movie] - train_vector[comm_movie]) ** 2)
        sim_score = 1.0/(euclidean_dist + 1)
        return sim_score
    den1 = math.sqrt(test_sum)
    den2 = math.sqrt(train_sum)
    if den1 and den2 != 0:
        cosine_similarity_weight = float(numerator/(den1*den2))
        #  TODO: Why cosine similarity weight is None?
        return cosine_similarity_weight


# find the movie ids for which ratings are to be predicted from test_matrix
def compute_cosine_similarity(test_vector,train_vector):

    comparable_movies = np.where(test_vector > 0)
    min_corrated = 2
    # if len(comparable_movies[0]) == 5:
    #     min_corrated = 2
    # elif len(comparable_movies[0]) == 10:
    #     min_corrated = 3
    # elif len(comparable_movies[0]) == 20:
    #     min_corrated = 4

    test_sum = 0.0
    train_sum = 0.0
    numerator = 0.0
    num_comm_movies = 0
    comm_movie = []
    for movie in comparable_movies[0]:
        if train_vector[movie] > 0:
            num_comm_movies += 1
            comm_movie.append(movie)
            test_sum += (test_vector[movie] ** 2)
            train_sum += (train_vector[movie] ** 2)
            numerator += (train_vector[movie] * test_vector[movie])

    distance = 0.0
    if num_comm_movies < min_corrated:
        sim_score = 0.0
        for movie in comm_movie:
            distance += ((test_vector[movie] - train_vector[movie]) ** 2)
        euclidean_dist = math.sqrt(distance)
        sim_score = 1.0/(euclidean_dist + 1)
        return sim_score
    # Euclidean distance calculation for users with one common movie - bcos cosine sim will be 1 for such cases
    # if num_comm_movies == 1:
    #     euclidean_dist = math.sqrt((test_vector[comm_movie] - train_vector[comm_movie]) ** 2)
    #     sim_score = 1.0/(euclidean_dist + 1)
    #     return sim_score
    den1 = math.sqrt(test_sum)
    den2 = math.sqrt(train_sum)
    if den1 and den2 != 0:
        cosine_similarity_weight = float(numerator/(den1*den2))
        return cosine_similarity_weight


def predict_rating_cosine_case_amp(top_k_weight_rating_list):
    denominator = 0.0
    numerator = 0.0
    for wt_rating in top_k_weight_rating_list:
        amplified_wt = wt_rating[0] * (wt_rating[0] ** 1.5)
        denominator += amplified_wt
        numerator += (amplified_wt * wt_rating[1])

    p_rating = float(numerator/denominator)
    return p_rating


def predict_rating_cosine(top_k_weight_rating_list):
    denominator = 0.0
    numerator = 0.0
    for wt_rating in top_k_weight_rating_list:
        denominator += wt_rating[0]
        numerator += (wt_rating[0] * wt_rating[1])

    p_rating = float(numerator/denominator)
    return p_rating


def calculate_rating_cosine(train_matrix, test_matrix, in_tuple, out_tuple):
    weight_list = []
    k = 102

    for i in range(len(train_matrix)):
        if train_matrix[i][in_tuple[2]] != 0:
            similar_user_rating = train_matrix[i][in_tuple[2]]
            # Calculate basic cosine similarity
            similar_user_weight = compute_cosine_similarity(test_matrix[in_tuple[1]], train_matrix[i])
            # Calculate iuf cosine similarity
            # similar_user_weight = compute_cosine_similarity_iuf(test_matrix[each_tuple[1]], train_matrix[i], iuf_dict)
            if similar_user_weight is not None:
                weight_list.append((similar_user_weight, similar_user_rating))
    weight_list.sort(key=lambda x: x[0], reverse=True)
    top_k_weight_list = weight_list[:k]
    if len(top_k_weight_list) != 0:
        p_rating = predict_rating_cosine(top_k_weight_list)
        # p_rating = predict_rating_cosine_case_amp(top_k_weight_list)
    else:
        p_rating = ((np.sum(test_matrix[in_tuple[1]]) * 1.0) / out_tuple[1])
    return p_rating


def calculate_rating_cosine_caseamp(train_matrix, test_matrix, in_tuple, out_tuple):
    weight_list = []
    k = 102

    for i in range(len(train_matrix)):
        if train_matrix[i][in_tuple[2]] != 0:
            similar_user_rating = train_matrix[i][in_tuple[2]]
            # Calculate normal cosine similarity
            similar_user_weight = compute_cosine_similarity(test_matrix[in_tuple[1]], train_matrix[i])
            # Calculate iuf cosine similarity
            # similar_user_weight = compute_cosine_similarity_iuf(test_matrix[each_tuple[1]], train_matrix[i], iuf_dict)
            if similar_user_weight is not None:
                weight_list.append((similar_user_weight, similar_user_rating))
    weight_list.sort(key=lambda x: x[0], reverse=True)
    top_k_weight_list = weight_list[:k]
    if len(top_k_weight_list) != 0:
        # p_rating = predict_rating_cosine(top_k_weight_list)
        p_rating = predict_rating_cosine_case_amp(top_k_weight_list)
    else:
        p_rating = ((np.sum(test_matrix[in_tuple[1]]) * 1.0) / out_tuple[1])
    return p_rating


def main():
    start_time = time.time()
    file_dict = {'test5.txt': ('result5.txt',5), 'test10.txt': ('result10.txt',10), 'test20.txt': ('result20.txt',20)}
    dir_path = '/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/'

    # Form a train_matrix from train.txt file -- train matrix is a 2d numpy array
    train_data = pd.read_csv("/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/train.txt", delimiter='\t', header=None)

    train_matrix = train_data.as_matrix()
    # iuf_train_matrix = iuf_calc(train_matrix)
    iuf_dict = iuf_calc(train_matrix)

    for in_file, out_tuple in file_dict.items():
        # test_matrix = np.zeros((100, 1000), int)
        test_input = os.path.join(dir_path, in_file)
        return_tuple = build_test_matrix(test_input)
        # print('return_tuple[0] :', return_tuple[0])
        test_matrix = return_tuple[0]
        # print('return_tuple[1] :', return_tuple[1])
        user_id_movie_id_list = return_tuple[1]
        result_file = os.path.join(dir_path, out_tuple[0])
        with open(result_file, 'w') as result:
            for in_tuple in user_id_movie_id_list:
                # print("each_tuple :", each_tuple)
                p_rating = calculate_rating_cosine_caseamp(train_matrix, test_matrix, in_tuple, out_tuple)
                p_rating = int(round(p_rating))
                result_string = in_tuple[0] + ' ' + str(in_tuple[2] + 1) + ' ' + str(p_rating)
                print("cosine :", result_string)
                result.write(result_string + "\n")
            result.close()
    print("Total Running Time : %s seconds " % (time.time() - start_time))

if __name__ == "__main__":
    main()