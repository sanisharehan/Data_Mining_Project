'''
utility.py: This file holds the functions commonly used by different classes throughout
the application.

Author: Sanisha Rehan
'''
import pandas as pd
import numpy as np
import math
import re
import io

from math import sqrt
from sklearn.metrics import mean_squared_error


#------------------------------------------------------------------------------
def generate_df_from_records(records_array, desired_columns=None):
    """
    Method to generate a pandas dataframe from a record array.
    """
    # Check if the passed records is not empty.
    if len(records_array) == 0:
        print ("Empty records passed. Cannot gerenate dataframe.")
        return None
    
    if not desired_columns:
        return pd.DataFrame.from_records(records_array)
    else:
        return pd.DataFrame.from_records(records_array, columns=desired_columns)

#------------------------------------------------------------------------------
def convert_string_from_unicode_to_ascii(ip_unicode):
    """
    """
    ip_value = ip_unicode.replace('\n', ' ')
    try:
        re.sub(r'[^\x00-\x7F]+',' ', ip_value)
        val = ip_value.encode('ascii','ignore')
    except Exception as e:
        print ("Exception in converting %s to ASCII: %s" % (ip_unicode, e))
        return
    return val

#------------------------------------------------------------------------------
def save_df_to_csv(ip_dataframe, csv_file_loc, add_index=False):
    """
    Function to save a given dataframe to csv file.
    """
    if len(ip_dataframe.index) == 0:
        print ("Dataframe empty. Not saving to csv file.")
        return
    ip_dataframe.to_csv(csv_file_loc, index=add_index)
    
#------------------------------------------------------------------------------
def read_csv_data_to_df(csv_file_loc, headers=True):
    """
    """
    return pd.read_csv(csv_file_loc)

#------------------------------------------------------------------------------
def round_to_closest_pt_5(ip_val):
    """
    Function to round a number to closest 0.5.
    """
    if math.isnan(ip_val):
        return 0.
    if math.isinf(ip_val):
        return 0.
    return round(ip_val * 2)/2

#------------------------------------------------------------------------------
def convert_df_to_np_array(ip_dataframe):
    """
    Utility to convert a given dataframe to np array without headers and index columns.
    """
    df = pd.read_csv(io.StringIO(u""+ip_dataframe.to_csv(header=None,index=False)), header=None)
    np_array = np.array(df)

    # Convert nans to 0
    np_array = np.nan_to_num(np_array)
    return np_array
    

#------------------------------------------------------------------------------
def calculate_rmse_model(actual_df, predicted_df, predicted_vals_map):
    """
    Function to calculate RMSE for input actual df, predicted df for given predicted
    values map. The map holds zipcodes and list of columns for which values
    are predicted for those zipcodes.
    """
    actual_values_array = []
    predicted_values_array = []
    actual_df_cols_list = actual_df.columns.tolist()
    
    '''
    for zipcode, col_list in predicted_vals_map.iteritems():
        # Get row corresponding to zipcode.
        act_row = actual_df[actual_df['zipcode'] == zipcode]
        pred_row = predicted_df[predicted_df['zipcode'] == zipcode]
        
        # Get values for columns predicted.
        for col_name in col_list:
            act_col_val = act_row[col_name].tolist()[0]
            pred_col_val = pred_row[col_name].tolist()[0]
            
            # Append these values to respective lists.
            actual_values_array.append(round_to_closest_pt_5(act_col_val))
            predicted_values_array.append(round_to_closest_pt_5(pred_col_val))
    '''
    for act_index, col_list in predicted_vals_map.iteritems():
        # First get the row from actual_df
        act_row = actual_df[actual_df.index == act_index]
        act_zipcode = act_row['zipcode'].tolist()[0]
        
        # Get the corresponding row from predicted df
        pred_row = predicted_df[predicted_df['zipcode'] == act_zipcode]
        
        # Get values corresponding to changed columns.
        for col_index in col_list:
            col_name = actual_df_cols_list[col_index]
            act_col_val = act_row[col_name].tolist()[0]
            pred_col_val = pred_row[col_name].tolist()[0] 
            
            # Append these values to respective lists.
            actual_values_array.append(act_col_val)
            predicted_values_array.append(pred_col_val)

            error = abs(act_col_val - pred_col_val)
            if error >= 2:
                print ("Error: %f, zipcode: %s, col: %s" % (error, str(act_zipcode), col_name))
    
    print "Actual array: ", actual_values_array
    print "Predicted array: ", predicted_values_array
    print "Number of values: ", len(actual_values_array)

    # Calculate RMSE using the two vectors.
    rmse = sqrt(mean_squared_error(np.array(actual_values_array), np.array(predicted_values_array)))
   
    gg = np.array(actual_values_array)
    hh = np.array(predicted_values_array)
    diff = gg - hh
    
    print ("Mean of error: %f, std deviation: %f" % (np.mean(diff), np.std(diff)))

    print "The RMSE is: ", rmse    
    return rmse

#------------------------------------------------------------------------------
