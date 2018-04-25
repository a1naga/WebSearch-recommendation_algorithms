import pandas as pd
import numpy as np
import os
import time
from recommendation_system_cosine import calculate_rating_cosine as basic_cosine_sim
from recommendation_system_cosine import calculate_rating_cosine_caseamp as cosine_sim_case_amp
from recommendation_system_pearson import calculate_rating as pearson

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


def main():
    start_time = time.time()
    file_dict = {'test5.txt': ('result5.txt',5), 'test10.txt': ('result10.txt',10), 'test20.txt': ('result20.txt',20)}
    dir_path = '/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/'
    # Form a train_matrix from train.txt file -- train matrix is a 2d numpy array
    train_data = pd.read_csv("/Users/aarthi/Winter2018-COEN272-Web_Search/project-2/train.txt", delimiter='\t', header=None)
    train_matrix = train_data.as_matrix()

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
                rating_basic_cosine = basic_cosine_sim(train_matrix, test_matrix, in_tuple, out_tuple)
                rating_cosine_caseamp = cosine_sim_case_amp(train_matrix, test_matrix, in_tuple, out_tuple)
                rating_pearson = pearson(train_matrix, test_matrix, in_tuple, out_tuple)

                # print("rating_basic_cosine :", rating_basic_cosine, "rating_cosine_caseamp :", rating_cosine_caseamp,  "rating_pearson :", rating_pearson)
                p_rating = int(round(((0.3*rating_basic_cosine) +(0.3*rating_cosine_caseamp) + (0.4*rating_pearson))/1.0))
                # print("p_rating :", p_rating)

                # write into result file
                result_string = in_tuple[0] + ' ' + str(in_tuple[2] + 1) + ' ' + str(p_rating)
                print("ensemble :", result_string)
                result.write(result_string + "\n")
            result.close()
    print("Total Running Time : %s seconds " % (time.time() - start_time))
if __name__ == "__main__":
    main()