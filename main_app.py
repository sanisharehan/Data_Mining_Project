'''
main_app.py: File from where the application starts.

Author: Sanisha Rehan
'''

from constants import YELP_BUSINESS_DATA_FILENAME
from data_filtering_engine import Data_Filtering_Engine
from data_preprocessing_engine import Data_Preprocessing_Engine
from data_pivoting_engine import Data_Pivoting_Aggregation_Engine
from data_split_engine import Training_And_Test_Data_Split
from recommender_engine import Recommender_Engine
from utility import save_df_to_csv, calculate_rmse_model

import datetime
import random
import io
import math

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
    #chaned_values_map = {}
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
            print ("Replacing index: %d, column: %s" %(index_sel, col_sel))
            test_data_to_return.ix[index_sel, (rnd_col+1)] = None
            total_values_changed_to_nan += 1
            
    print ("Total values changed to NaN: %d" %(total_values_changed_to_nan))
    return test_data_to_return, changed_indices_map   
    

#------------------------------------------------------------------------------
def perform_training_recommendation(train_df):
    """
    """
    rec_model_obj = Recommender_Engine(train_df.copy(), 30, False)
    predicted_train_result_df = rec_model_obj.get_predicted_ratings_df()
    
    # Save results to a file (optional)
    save_df_to_csv(predicted_train_result_df, "yelp_train_data_predicted.csv")


#------------------------------------------------------------------------------
def perform_test_recommendation(test_df):
    """
    """
    # Generate test dataset with randomly allocated Nan in the given df.
    test_df_for_eval, missing_ratings_map = generate_test_data_with_missing_values(test_df.copy(), 100)
    rec_model_obj = Recommender_Engine(test_df_for_eval.copy(), 30, False)

    predicted_test_result_df = rec_model_obj.get_predicted_ratings_df()

    #missing_ratings_map = rec_model_obj.get_missing_columns_map()

    # Calculate RMSE between actual and predicted values of test data.
    calculate_rmse_model(test_df, predicted_test_result_df, missing_ratings_map)


#------------------------------------------------------------------------------
def start_data_preprocessing(ip_file_name):
    """
    Function that performs step by step operations for filtering, preprocessing of
    input data.
    """
    # Step 1: Perform filtering on data.
    data_filter_obj = Data_Filtering_Engine(ip_file_name)
    filtered_df = data_filter_obj.get_final_dataframe()
    
    # Step 2: Generate the dataframe with business categories mapped.
    data_preprocess_obj = Data_Preprocessing_Engine(filtered_df.copy())
    preprocessed_df = data_preprocess_obj.get_final_preprocessed_df()

    # Step 3: Save the results to an output file.
    save_df_to_csv(preprocessed_df, "yelp_preprocessed_data.csv")

    # Step 4: Get pivoted data.
    data_pivot_engine = Data_Pivoting_Aggregation_Engine(preprocessed_df.copy())
    pivoted_df = data_pivot_engine.get_pivoted_df()

    return pivoted_df    


#------------------------------------------------------------------------------
def generate_training_and_test_data(ip_pivot_df):
    """
    Function for generating test and training data.
    """
    data_split_obj = Training_And_Test_Data_Split(ip_pivot_df.copy(), 20)
    training_df = data_split_obj.get_train_dataframe()
    test_df = data_split_obj.get_test_dataframe()

    return training_df, test_df


#------------------------------------------------------------------------------
# Main function for staring the application.
if __name__ == '__main__':
    ip_file_name = YELP_BUSINESS_DATA_FILENAME
    
    # Data preprocessing and get pivot df.
    pivot_df = start_data_preprocessing(ip_file_name)

    # Generate test and training df.
    training_df, test_df = generate_training_and_test_data(pivot_df.copy())

    # Perform recommendation on training data.
    perform_training_recommendation(training_df.copy())

    # Preform recommendation on test data.
    perform_test_recommendation(test_df.copy())
    print "Done"
