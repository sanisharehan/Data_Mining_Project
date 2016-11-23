'''
data_split_engine.py: File with class defined for splitting input data between test
and training datasets.

Author: Sanisha Rehan
'''
import pandas as pd
import random
import datetime

class Training_And_Test_Data_Split(object):
    """
    Class that performs the split between training and test data for the Yelp pivoted data.
    """
    def __init__(self, ip_pivot_dataframe, test_data_split_pct):
        self._ip_dataframe = ip_pivot_dataframe
        self._test_data_df = None
        self._train_data_df = None
        self._test_data_split_pct = test_data_split_pct
        self.generate_test_train_data()
        
    def generate_test_data(self):
        # Randomly select the columns and zipcodes from given data
        # to create a test set.
        # Get rows which have all filled values.
	# We can pick up some rows randomly from the training set.
        all_indices_list = self._ip_dataframe.index.tolist()
        len_index_list = len(all_indices_list)
        num_test_rows = (len_index_list * self._test_data_split_pct/100)
        
        random.seed(datetime.datetime.now())
        test_df_rows_idx = []
        for i in range(0, num_test_rows):
            rnd_index = random.randint(0, len_index_list - 1)
            index = all_indices_list[rnd_index]
            if index not in test_df_rows_idx:
                test_df_rows_idx.append(index)
                
        # Check total rows to be added to test df.
        print ("Total rows to be added to test df: %d" % len(test_df_rows_idx))
        
        # Create test df with those rows
        test_data_initial = self._ip_dataframe[self._ip_dataframe.index.isin(test_df_rows_idx)]
        self._test_data_df = test_data_initial.copy()
        #test_data_initial = self._ip_dataframe[~pd.isnull(self._ip_dataframe).any(axis=1)]
        #self._test_data_df = test_data_initial.copy()
        
        
    def generate_train_data(self):
        """
        """
        #train_data_initial = self._ip_dataframe[~self._ip_dataframe.index.isin(self._test_data_df.index)]
	train_data_initial = self._ip_dataframe.copy()
        self._train_data_df = train_data_initial.copy()
        
    
    def get_test_dataframe(self):
        """
        """
        return self._test_data_df
    
    def get_train_dataframe(self):
        """
        """
        return self._train_data_df
    
    def generate_test_train_data(self):
        """
        """
        print ("LOG: [Test Train Data Engine] Generating test-train data.")
        self.generate_test_data()
        self.generate_train_data()
        print ("LOG: [Test Train Data Engine] Stats:")
        print ("Rows (Original data): %d" % len(self._ip_dataframe.index))
        print ("Rows (Test data): %d" % len(self._test_data_df.index))
        print ("Rows (Training data): %d" % len(self._train_data_df.index))
        print ("LOG: [Test Train Data Engine] Generated test-train data.")
