'''
data_filtering_engine.py: THis file holds the class created with various methods
used to perform initial filtering on Yelp business review data.

Author: Sanisha Rehan
'''
from constants import YELP_DESIRED_COLUMNS_LIST
from uszipcode import ZipcodeSearchEngine
from utility import generate_df_from_records

import json
import numpy as np
import pandas as pd
import re

class Data_Filtering_Engine(object):
    """
    Class for performing data cleanup and filtering of Yelp business review data.
    """
    def __init__(self, filename):
        self._file_name = filename
        self._input_data_records = list()
        self._input_filtered_cols_records = list()
        self._input_dataframe = None
        self.perform_data_filtering_and_cleanup()
        
    def read_ip_json_file_to_list(self):
        """
        Reads input json file and return list of records.
        """
        ip_data = []
        with open(self._file_name) as f:
            for line in f:
                a = json.loads(line)
                ip_data.append(a)
        f.close()
        print ("LOG: [Filtering Engine] Read input file.")
        print ("LOG: [Filtering Engine] Number of records fetched: %d" % len(ip_data))
        self._input_data_records = ip_data
        
    def filter_desired_columns_from_ip_records(self):
        """
        Filter only desired columns from the records. 
        """
        filtered_data = []
        
        print("LOG: [Filtering Engine] Filtering desired columns.")
        # Get categories and convert to list.
        for row in self._input_data_records:
            cat_list = []
            for k in row['categories']:
                cat_list.append(k.encode('ascii'))
            
            # Parse zipcode from the full_address value.
            zip_code = row['full_address'].split(' ')[len(row['full_address'].split(' ')) - 1]
    
            # Check if zipcode is available and a valid one.
            try:
                zip_code = int(zip_code)
                # Sometimes we get invalid zipcode such as 891118, we need to get 
                # the zipcode from latitude and longitude
                if (zip_code > 99999):
                    raise Exception("ERROR: [Filtering Engine] Invalid zip_code")
            except:
                # Get the closest zipcode for the given lat-long
                # Help link: https://pypi.python.org/pypi/uszipcode
                # Search engine for zipcode to lat-long and vice-versa conversions. This returns
                # top 3 matching zipcodes.
                search = ZipcodeSearchEngine()
                result = search.by_coordinate(row['latitude'], 
                                              row['longitude'], 
                                              radius=20, 
                                              returns=3)
                if len(result) == 0:
                    continue
                zip_code = result[0]['Zipcode']
                
            # Filter out rows that belong to some invalid locations.
            if (zip_code < 100):
                continue
        
            # Create record row with desired columns.
            a = (cat_list, '', 
                 row['state'], row['city'], 
                 row['full_address'], zip_code, 
                 row['longitude'], row['latitude'], 
                 row['stars'], row['type'], 
                 row['review_count']
                )
            
            # Append to final data.
            filtered_data.append(a)
            
        print ("LOG: [Filtering Engine] Number of filtered final records: %d" % len(filtered_data))
        self._input_filtered_cols_records = filtered_data
        
    def convert_records_to_df(self):
        """
        Function to convert records to a Dataframe.
        """
        print ("LOG: [Filtering Engine] Converting records to dataframe.")
        self._input_dataframe = generate_df_from_records(
            self._input_filtered_cols_records, 
            YELP_DESIRED_COLUMNS_LIST
        )
        print ("LOG: [Filtering Engine] Converted records to dataframe.")
        
    def get_final_dataframe(self):
        """
        Function returns generated pre-processed dataframe.
        """
        return self._input_dataframe
    
    def perform_data_filtering_and_cleanup(self):
        """
        Step by step function calling to filter and cleanup the yelp business review data.
        """
        print ("LOG: [Filtering Engine] Performing data filtering.")
        self.read_ip_json_file_to_list()
        self.filter_desired_columns_from_ip_records()
        self.convert_records_to_df()
        print ("LOG: [Filtering Engine] Performed data filtering.")
