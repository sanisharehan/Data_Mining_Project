'''
main_app.py: File from where the application starts.
===========
How to Run:
===========

a. Run with default options:
python main_app.py

b. Run with different options:
python main_app.py -r train -k 15 -s cosine

Various options:
    a. To specify the running dataset for recommender system. 
    Default value: all
    -r <string>
    -r all
    -r train
    -r test

    b. To specify number of nearest neighbours for User based collaborative filtering.
    Default value: 20
    -k <int>

    c. Similarity to be used for collaborative filtering.
    Default value: cosine
    -s cosine/geo

    d. Specify the type of test method used.
    Default value: fast
    -tm fast/all

    fast- Uses randomly chosen cells from the entire data, computes the rating and
    finds error in predicted and actual values.

    all- Selects one row from test data, append to training data, computes the rating
    for a given business type and finds the error.

Author: Sanisha Rehan
'''

from constants import YELP_BUSINESS_DATA_FILENAME, \
                        VALID_RUN_MODES,\
                        VALID_TEST_RUN_MODES,\
                        VALID_SIMILARITY_TYPES,\
                        FILTERED_FILENAME,\
                        PREPROCESSED_FILENAME,\
                        PIVOT_FILENAME,\
                        TRAINING_FILENAME,\
                        TRAINING_PRED_FILENAME,\
                        TEST_FILENAME,\
                        TEST_PRED_FILENAME
from data_filtering_engine import Data_Filtering_Engine
from data_preprocessing_engine import Data_Preprocessing_Engine
from data_pivoting_engine import Data_Pivoting_Aggregation_Engine
from data_split_engine import Training_And_Test_Data_Split
from recommender_engine import Recommender_Engine
from utility import save_df_to_csv, calculate_rmse_model, read_csv_data_to_df, plot_error_histogram
from uszipcode import ZipcodeSearchEngine
from math import sqrt
from sklearn.metrics import mean_squared_error


import datetime
import random
import io
import math
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt


FINAL_RATINGS_DF = None 

#------------------------------------------------------------------------------
def get_ratings_for_business_zipcode(business_type, zipcode):
    """
    """
    # Get all zipcodes avl in the result.
    global FINAL_RATINGS_DF
    FINAL_RATINGS_DF = read_csv_data_to_df(TRAINING_PRED_FILENAME)
    print len(FINAL_RATINGS_DF)
    zipcode_list = np.array(FINAL_RATINGS_DF.zipcode).tolist()
    if zipcode in zipcode_list:
        rating_row = FINAL_RATINGS_DF[FINAL_RATINGS_DF['zipcode'] == zipcode]
        rating = rating_row[business_type].tolist()[0]
        print ("Predicted Rating for business: %s, zipcode: %d is %f" % (business_type, zipcode, rating)) 
        return rating
    else:
        search = ZipcodeSearchEngine()
        lat_long_inf = search.by_zipcode(str(zipcode))
        lat, longi = lat_long_inf["Latitude"], lat_long_inf["Longitude"]
        
        try:
            result = search.by_coordinate(lat, longi, radius=radius, returns=k_neigh)
        except:
            return None
        
        if len(result) == 0:
            return None
        else:
            nearest_zip_list = []
            for res in result:
                nearest_zip_list.append(int(res["Zipcode"]))
                
            # Check which all zipcodes are present in the given data.
            avl_zipcode = set(nearest_zip_list) & set(self._zip_code_list)
            if avl_zipcode is not None:
                avl_zipcode_list = list(avl_zipcode)
                ratings = FINAL_RATINGS_DF[FINAL_RATINGS_DF['zipcode'].isin(avl_zipcode_list)]
                # Calculate avg rating.
                rating = 0
                for row in ratings.iterrows():
                    rating += ratings[business_type].tolist()[0]
                avg_rating = rating/len(ratings)
                print ("Predicted Rating for business: %s, zipcode: %d is %f" % (business_type, zipcode, avg_rating)) 
            else:
                return None

#------------------------------------------------------------------------------
def generate_test_data_with_missing_values(ip_test_df, num_nans):
    """
    Function to generate some random Nan in test data for model validation.
    """
    total_values_changed_to_nan = 0
    test_indices_list = ip_test_df.index.tolist()
    changed_indices_map = {}
    
    # Get columns except zipcode
    all_cols = ip_test_df.columns.tolist()
    ip_columns_list = all_cols[1:]
    
    len_index_list = len(test_indices_list)
    len_columns_list = len(ip_columns_list)
    
    # Final test data
    test_data_to_return = ip_test_df.copy()
    
    # Set seed value
    random.seed(datetime.datetime.now())
    for i in range(0, num_nans):
        rnd_index = random.randint(0, len_index_list-1)
        rnd_col = random.randint(0, len_columns_list-1)

        # Select index and column
        index_sel = test_indices_list[rnd_index]
        col_sel = ip_columns_list[rnd_col]

        # Get to the row and column and replace value in copy dataframe with nan
        col_value = test_data_to_return.ix[index_sel, (rnd_col+1)]
        if not math.isnan(col_value):
            # If the index-col pair is not in the map, add
            if index_sel in changed_indices_map.keys():
                if rnd_col in changed_indices_map[index_sel]:
                    pass
                else:
                    changed_indices_map[index_sel].append(rnd_col+1)
            else:
                changed_indices_map[index_sel] = [rnd_col+1]
            
            # Change the value of that cell to None
            test_data_to_return.ix[index_sel, (rnd_col+1)] = None
            total_values_changed_to_nan += 1
            
    print ("Total values changed to NaN: %d" %(total_values_changed_to_nan))
    return test_data_to_return, changed_indices_map   
    
#------------------------------------------------------------------------------
def perform_fast_test_recommendation(test_df, k_neigh, similarity, plot_graph):
    """
    For the fast test experimentation, cells are randomly chosen from the entire
    training data, their values are stored and then converted to Nans. The new
    values are computed and compared against the actual values.

    We get the results in terms of RMSE, Mean of difference, Standard Deviation
    of the difference.
    """
    print ("LOG: [Main] Running recommendation engine for test data in 'FAST' mode.")
    
    # Generate test dataset with randomly allocated Nan in the given df.
    total_cells = test_df.shape[0] * test_df.shape[1]
    num_test_cells = int(0.1 * total_cells)
    test_df_for_eval, predicted_ratings_map = generate_test_data_with_missing_values(test_df.copy(), num_test_cells)

    geo_similarity = False
    if similarity == 'geo':
        geo_similarity = True
    rec_model_obj = Recommender_Engine(test_df_for_eval.copy(), k_neigh, geo_similarity)
    
    predicted_test_result_df = rec_model_obj.get_predicted_ratings_df()

    # Save actual and predicted results into files.
    actual_data_filename = "yelp_test_data_fast_actual.csv"
    predicted_data_filename = "yelp_test_data_fast_predicted.csv"
    save_df_to_csv(test_df, actual_data_filename)
    save_df_to_csv(predicted_test_result_df, predicted_data_filename)

    print("LOG: [Main] Saved actual test data for 'Fast' mode into file '%s'" % actual_data_filename)
    print("LOG: [Main] Saved predicted test data for 'Fast' mode into file '%s'" % predicted_data_filename)

    # Calculate RMSE between actual and predicted values of test data.
    rmse = calculate_rmse_model(test_df, predicted_test_result_df, predicted_ratings_map, plot_graph)

#------------------------------------------------------------------------------
def perform_step_by_step_test_recommendation(test_df, train_df, k_neigh, similarity, plot_graph):
    """
    This method takes one test row with actual rating value saved, appends the row to
    the training data and predicts the rating. All test rows are added one by
    one into the training set.

    Finally, actual values are compared against predicted values.
    """
    print ("LOG: [Main] Running recommendation engine for test data in 'STEP By STEP' mode.")
    # Get all columns.
    all_cols_list = test_df.columns.tolist()

    # Remove all test rows from training data.
    train_data_copy = train_df[~train_df.index.isin(test_df.index)].copy()

    # Generate test dataset with randomly allocated Nan in the given df.
    total_cells = test_df.shape[0] * test_df.shape[1]
    test_cells = int(0.1 * total_cells)
    test_df_for_eval, changed_ratings_map = generate_test_data_with_missing_values(test_df.copy(), test_cells)

    actual_values_array = []
    predicted_values_array = []

    # Value of similarity.
    geo_similarity = False
    if similarity == 'geo':
        geo_similarity = True

    for row_index, cols_changed in changed_ratings_map.iteritems():
        # Test row
        test_row = test_df_for_eval[test_df_for_eval.index == row_index]
        
        # Actual row
        actual_row = test_df[test_df.index == row_index]

        # Append this row to training set and evaluate the value for missing column.
        new_train_data = train_data_copy.append(test_row).copy()

        # Pass this new data to the engine.
        rec_model_obj = Recommender_Engine(new_train_data.copy(), k_neigh, geo_similarity)
        predicted_test_result_df = rec_model_obj.get_predicted_ratings_df()

        test_zipcode = test_row['zipcode'].tolist()[0]
        pred_row = predicted_test_result_df[predicted_test_result_df['zipcode'] == test_zipcode]
        pred_row_index = pred_row.index.tolist()[0]
        
        for col_index in cols_changed:
            col_name = all_cols_list[col_index]
            pred_val = pred_row.loc[pred_row_index, col_name]
            act_val = actual_row.loc[row_index, col_name]

            actual_values_array.append(act_val)
            predicted_values_array.append(pred_val)
        print "\n"

    # Calculate error
    act_np = np.array(actual_values_array)
    pred_np = np.array(predicted_values_array)
    diff = act_np - pred_np
    rmse = sqrt(mean_squared_error(act_np, pred_np))

    print ("LOG: [Main] Number of values predicted: %d" % len(actual_values_array))
    print ("LOG: [Main] Mean Error: %f, Standard Deviation: %f, RMSE: %d" %(np.mean(diff), np.std(diff), rmse))

    # Plot graph.
    if plot_graph:
        plot_error_histogram(diff)

#------------------------------------------------------------------------------
def perform_training_recommendation(train_df, k_neigh, similarity):
    """
    Run recommendation on entire training data and save final results for
    web app.
    """
    global FINAL_RATINGS_DF
    print ("LOG: [Main] Running recommendation engine for training data.")
    geo_similarity = False
    if similarity == 'geo':
        geo_similarity = True

    rec_model_obj = Recommender_Engine(train_df.copy(), k_neigh, geo_similarity)
    predicted_train_result_df = rec_model_obj.get_predicted_ratings_df()

    FINAL_RATINGS_DF = predicted_train_result_df.copy()
    
    # Save results to a file (optional)
    save_df_to_csv(predicted_train_result_df, TRAINING_PRED_FILENAME)
    print("LOG: [Main] Saved predicted training results in file '%s'" % TRAINING_PRED_FILENAME)

#------------------------------------------------------------------------------
def generate_training_and_test_data(ip_pivot_df):
    """
    Function for generating test and training data.
    """
    data_split_obj = Training_And_Test_Data_Split(ip_pivot_df.copy(), 10)
    training_df = data_split_obj.get_train_dataframe()
    test_df = data_split_obj.get_test_dataframe()

    save_df_to_csv(training_df, TRAINING_FILENAME)
    save_df_to_csv(test_df, TEST_FILENAME)
    print ("LOG: [Main] Saved training data in file '%s'" % TRAINING_FILENAME)
    print ("LOG: [Main] Saved test data in file '%s'" % TEST_FILENAME)

    return training_df, test_df

#------------------------------------------------------------------------------
def start_data_preprocessing(ip_file_name):
    """
    Function that performs step by step operations for filtering, preprocessing of
    input data.
    """
    # Step 1: Perform filtering on data.
    data_filter_obj = Data_Filtering_Engine(ip_file_name)
    filtered_df = data_filter_obj.get_final_dataframe()
    save_df_to_csv(filtered_df, FILTERED_FILENAME)
    print ("LOG: [Main] Saved filtered results into file '%s'\n" % FILTERED_FILENAME)
    
    # Step 2: Generate the dataframe with business categories mapped.
    data_preprocess_obj = Data_Preprocessing_Engine(filtered_df.copy())
    preprocessed_df = data_preprocess_obj.get_final_preprocessed_df()
    save_df_to_csv(preprocessed_df, PREPROCESSED_FILENAME)
    print ("LOG: [Main] Saved preprocessed results into file '%s'\n" % PREPROCESSED_FILENAME)

    # Step 3: Get pivoted data.
    data_pivot_engine = Data_Pivoting_Aggregation_Engine(preprocessed_df.copy())
    pivoted_df = data_pivot_engine.get_pivoted_df()
    save_df_to_csv(pivoted_df, PIVOT_FILENAME)
    r = pivoted_df['zipcode'].value_counts()
    print ("Len of pivoted data: ", pivoted_df.shape[0])
    for k, v in r.iteritems():
        if v > 1:
            print k
    print ("LOG: [Main] Saved pivoted results into file '%s'\n" % PIVOT_FILENAME)

    return pivoted_df    

#------------------------------------------------------------------------------
def perform_recommendation(run_mode, k_neigh, similarity, test_mode, plot_graph, read_file):
    """
    """
    pivot_df = None
    if not read_file:
        # Step 1: Initial preprocessing and pivoting steps. 
        ip_file_name = YELP_BUSINESS_DATA_FILENAME
        # Data preprocessing and get pivot df.
        pivot_df = start_data_preprocessing(ip_file_name)
    else:
        # Read the pivoted data from file.
        print ("LOG: [Main] Reading pivoted data from file '%s'" % PIVOT_FILENAME)
        pivot_df = read_csv_data_to_df(PIVOT_FILENAME)

    r = pivot_df['zipcode'].value_counts()
    for k, v in r.iteritems():
        if v > 1:
            print k
    # Generate test and training df.
    training_df, test_df = generate_training_and_test_data(pivot_df.copy())

    # Step 2: Perform recommendation.
    if run_mode == 'all':
        # Run the recommender system for both training and test datasets.
        # a. Perform recommendation on training data.
        perform_training_recommendation(training_df.copy(), k_neigh, similarity)
        print

        # b. Perform recommendation on test data.
        if test_mode == 'fast':
            perform_fast_test_recommendation(training_df.copy(), k_neigh, similarity, plot_graph)
        elif test_mode == 'all':
            perform_step_by_step_test_recommendation(test_df.copy(), 
                                                    training_df.copy(), 
                                                    k_neigh, 
                                                    similarity,
                                                    plot_graph
                                                    )
    # Only run for training data.
    elif run_mode == 'train':
        perform_training_recommendation(training_df.copy(), k_neigh, similarity)
    # ONly run for test data.
    elif run_mode == 'test':
         if test_mode == 'fast':
            perform_fast_test_recommendation(training_df.copy(), k_neigh, similarity, plot_graph)
         elif test_mode == 'all':
            perform_step_by_step_test_recommendation(test_df.copy(), 
                                                    training_df.copy(), 
                                                    k_neigh, 
                                                    similarity,
                                                    plot_graph
                                                    )
    
#------------------------------------------------------------------------------
def start_application(args):
    """
    Runs application based on the input parameters passed.
    
    Various options:
    a. To specify the running dataset for recommender system. 
    Default value: all
    -r <string>
    -r all
    -r train
    -r test

    b. To specify number of nearest neighbours for User based collaborative filtering.
    Default value: 20
    -k <int>

    c. Similarity to be used for collaborative filtering.
    Default value: cosine
    -s cosine/geo

    d. Specify the type of test method used.
    Default value: fast
    -m fast/all

    fast- Uses randomly chosen cells from the entire data, computes the rating and
    finds error in predicted and actual values.

    all- Selects one row from test data, append to training data, computes the rating
    for a given business type and finds the error.

    e. To plot the histogram for error in actual and predicted data for test data.
    -p
    Plot True

    f. To read the pivoted data from saved file.
    -f
    Read from file True.
    """
    help_text = "python main_app.py -r <dataset> -k <neighbors> -s <similarity> -tm <type_of_testing>"
    try:
        opts, remainder = getopt.getopt(args, "hr:k:s:m:pf")
    except getopt.GetoptError as e:
        print help_text
        sys.exit(2)

    # Read all arguments.
    run_mode = 'train'
    k_neigh = 20
    similarity_type = 'cosine'
    test_mode = 'fast'
    plot_graph = False
    read_file = False

    for opt, arg in opts:
        if opt == '-h':
            print help_text
            sys.exit()
        # Run mode.
        elif opt == '-r':
            run_mode = arg
            if run_mode not in VALID_RUN_MODES:
                print ("ERROR: [Main] Supported run modes are: ", VALID_RUN_MODES)
                sys.exit(1)
        # K-nearest neighbour
        elif opt == '-k':
            k_neigh = int(arg)
            if k_neigh < 0:
                print ("ERROR: [Main] Number of neighbours cannot be negative.")
                sys.exit(1)
        # Similarity/Nearest neighbour to use.
        elif opt == '-s':
            similarity_type = arg
            if similarity_type not in VALID_SIMILARITY_TYPES:
                print ("ERROR: [Main] Supported similarity types are: ", VALID_SIMILARITY_TYPES)
                sys.exit(1)
        # Test run mode
        elif opt == '-m':
            test_mode = arg
            if test_mode not in VALID_TEST_RUN_MODES:
                print test_mode
                print ("ERROR: [Main] Supported test modes are: ", VALID_TEST_RUN_MODES)
                sys.exit(1)
        elif opt == '-p':
            plot_graph = True
        elif opt == '-f':
            read_file = True
        else:
            print ("ERROR: [Main] Option %s no supported" % opt)
            sys.exit(1)

    # Print the values passed from terminal.
    print ("LOG: [Main] Run mode: '%s', Number of nearest neighbours: '%d', Similarity Type: '%s', "
            "Test mode: '%s', Plot Graph: '%d', Read From File: '%d'" %(run_mode,
            k_neigh, similarity_type, test_mode, plot_graph, read_file))

    # Run the application based on options passed.
    perform_recommendation(run_mode, k_neigh, similarity_type, test_mode, plot_graph, read_file)


#------------------------------------------------------------------------------
# Main function for staring the application.
if __name__ == '__main__':
    print ("LOG: [Main] Starting the recommendation application.")
    start_application(sys.argv[1:])
    print("LOG: [Main] Finished the recommendation application.")
