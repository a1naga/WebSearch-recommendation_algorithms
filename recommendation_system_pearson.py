import pandas as pd
import numpy as np
import math
import os
import time

# Form a test_matrix from test.txt file
# list of tuple(user_id,test_row_index,movie_id)
def build_test_matrix(test_input):
    test_matrix = np.zeros((100, 1000), int)
    with open(test_input) as test:
        # list of tuple(user_id,test_row_index,movie_id)
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
    return (test_matrix,user_id_movie_id_list)

def pearson_correlation(test_vector, train_vector, to_be_predicted_movie):
    test_user_rated_movies = np.where(test_vector > 0)
    min_corrated = 2
    test_user_avg_rating = (np.sum(test_vector) * 1.0) /(len(test_user_rated_movies[0]))
    train_user_rated_movies = np.where(train_vector > 0)
    train_user_avg_rating = ((np.sum(train_vector) - train_vector[to_be_predicted_movie]) * 1.0)\
                            /(len(train_user_rated_movies[0]) - 1)

    test_sum = 0.0
    train_sum = 0.0
    numerator = 0.0
    num_comm_movies = 0

    for movie in test_user_rated_movies[0]:
        if train_vector[movie] > 0:
            num_comm_movies += 1
            test_sum += ((test_vector[movie] - test_user_avg_rating) ** 2)
            train_sum += ((train_vector[movie] - train_user_avg_rating) ** 2)
            numerator += ((train_vector[movie] - train_user_avg_rating) * (test_vector[movie] - test_user_avg_rating))

    if num_comm_movies < min_corrated:
        return (0.0,test_user_avg_rating, 0)
    den1 = math.sqrt(test_sum)
    den2 = math.sqrt(train_sum)
    if den1 != 0 and den2 != 0:
        pearson_weight = numerator / (den1 * den2)
        return (pearson_weight, test_user_avg_rating, train_user_avg_rating)

def predict_rating_pearson(top_k_weight_rating_list):
    denominator = 0.0
    numerator = 0.0
    test_user_avg_rating = top_k_weight_rating_list[0][3]

    for wt_rating in top_k_weight_rating_list:
        denominator += abs(wt_rating[1])
        numerator += (wt_rating[1] * (wt_rating[2] - wt_rating[4]))

    if denominator == 0:
        p_rating = test_user_avg_rating
    else:
        p_rating = test_user_avg_rating + (numerator/denominator)

    return p_rating

def calculate_rating(train_matrix, test_matrix, in_tuple, out_tuple):
    weight_list = []
    k = 102

    for i in range(len(train_matrix)):
        if train_matrix[i][in_tuple[2]] != 0:
            to_be_predicted_movie = in_tuple[2]
            similar_user_rating = train_matrix[i][in_tuple[2]]
            pearson_tuple = pearson_correlation(test_matrix[in_tuple[1]], train_matrix[i], to_be_predicted_movie)
            # print("pearson_tuple :", pearson_tuple)
            similar_user_weight = pearson_tuple
            if similar_user_weight is not None and similar_user_weight[0] != 0:
                weight_list.append(
                    (abs(similar_user_weight[0]),
                     similar_user_weight[0],
                     similar_user_rating,
                     similar_user_weight[1],
                     similar_user_weight[2]))
    weight_list.sort(key=lambda x: x[0], reverse=True)
    top_k_weight_list = weight_list[:k]
    if len(top_k_weight_list) != 0:
        p_rating = predict_rating_pearson(top_k_weight_list)
    else:
        p_rating = ((np.sum(test_matrix[in_tuple[1]]) * 1.0) / out_tuple[1])
    # Rounding up the p_rating for 0 and negative values. Also for va;ues greater than 5
    if p_rating > 5:
        p_rating = 5
    elif p_rating <= 0:
        p_rating = 1

    return p_rating

def main():
    start_time = time.time()
    file_dict = {'test5.txt': ('result5.txt',5), 'test10.txt': ('result10.txt',10), 'test20.txt': ('result20.txt',20)}
    test_input_files = ['test5.txt','test10.txt','test20.txt']
    dir_path = '/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/'
    # Form a train_matrix from train.txt file -- train matrix is a 2d numpy array
    train_data = pd.read_csv("/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/train.txt", delimiter='\t', header=None)
    train_matrix = train_data.as_matrix()

    for in_file, out_tuple in file_dict.items():
        test_input = os.path.join(dir_path, in_file)
        return_tuple = build_test_matrix(test_input)
        test_matrix = return_tuple[0]
        user_id_movie_id_list = return_tuple[1]
        result_file = os.path.join(dir_path, out_tuple[0])
        with open(result_file, 'w') as result:
            for in_tuple in user_id_movie_id_list:
                p_rating = calculate_rating(train_matrix, test_matrix, in_tuple, out_tuple)
                p_rating = int(round(p_rating))

                if p_rating > 5:
                    p_rating = 5
                elif p_rating < 1:
                    p_rating = 1

                result_string = in_tuple[0] + ' ' + str(in_tuple[2] + 1) + ' ' + str(p_rating)

                print("pearson :", result_string)
                result.write(result_string + "\n")
            result.close()
    print("Total Running Time : %s seconds " % (time.time() - start_time))

if __name__ == "__main__":
    main()
