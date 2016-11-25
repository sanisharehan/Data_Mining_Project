'''
recommender_engine.py: This file provides the class for recommender engine built
using collaborative filtering.

Author: Sanisha Rehan
'''
from sklearn.metrics.pairwise import cosine_similarity
from utility import convert_df_to_np_array, round_to_closest_pt_5, generate_df_from_records, save_df_to_csv
from uszipcode import ZipcodeSearchEngine

import numpy as np
import pandas as pd
import io
import math

class Recommender_Engine(object):
    """
    Class for creating the recommender engine for data mining project.
    """
    def __init__(self, ip_dataframe, k_neigh=None, geographical_similarity=True):
        self._ip_dataframe = ip_dataframe.copy()
        self._k_neigh = k_neigh
        self._geographical_similarity = geographical_similarity

        self._all_cols = self._ip_dataframe.columns.tolist()
        
        self._zip_code_list = None
        self._similarity_matrix = None
        self._non_zip_cols = None
        self._df_matrix_array = None
        self._missing_ratings_map = None
        
        self._predicted_ratings_records = None
        
        self.perform_recommendation()
        
        
    def generate_row_similarities_matrix(self, ip_df):
        """
        Function to compute similarity matrix for rows provided in given
        df.
        """
        print ("LOG: [Recommender Engine] Generating similarity matrix.")
        sim_matrix = cosine_similarity(ip_df)
        self._similarity_matrix = sim_matrix
        sim_matrix.tofile("Similarity_matrix.csv", sep=",")
        print ("LOG: [Recommender Engine] Saved computed similarity matrix into file 'Similarity_matrix.csv")
        print ("LOG: [Recommender Engine] Generated similarity matrix.")
        
    
    def convert_ip_df_to_np_array(self):
        """
        Function to convert input dataframe to numpy array matrix to
        perform similarity calculations.
        """
        print ("LOG: [Recommender Engine] Generating matrix from df.")
        # Save zipcodes present in the ip data.
        self._zip_code_list = np.array(self._ip_dataframe['zipcode']).tolist()
        print ("LOG: [Recommender Engine] Number of zipcodes saved: %d" % len(self._zip_code_list))
        
        # Get all columns except zipcode.
        df_cols = self._ip_dataframe.columns.tolist()
        self._non_zip_cols = df_cols[1:]
        
        # Copy all columns except zipcode
        df_copy = self._ip_dataframe[self._non_zip_cols].copy()
        df_array = convert_df_to_np_array(df_copy)
        self._df_matrix_array = df_array
        print ("LOG: [Recommender Engine] Generated matrix from df.")
    
            
    def get_k_nearest_zipcodes_locations(self, ip_zipcode, radius=50, k_neigh=20):
        """
        Find the zipcodes near to the provided zipcodes.
        """
        search = ZipcodeSearchEngine()
        lat_long_inf = search.by_zipcode(str(ip_zipcode))
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
                zip_index_list = []
                for code in avl_zipcode:
                    zip_index_list.append(self._zip_code_list.index(code))
                return zip_index_list
            else:
                return None
    
    def get_top_n_similar_indices_and_similarities(self, 
                                                   similarity_matrix,
                                                   query_zipcode,
                                                   query_index,
                                                   k_neigh=None,
                                                   geographical_similarity=False):
        """
        Function to get top K rows similar to the row at given query index.
        """
        # Get row from similarity matrix corresponding to the query index
        similarity_vector = similarity_matrix[query_index]
        
        if k_neigh == None:
            sim_map = {k+1 : val for k, val in enumerate(similarity_vector[1:])}
            return sim_map

        # Case for geographical similarity.
        if geographical_similarity:
            k_nearest_zip_list = self.get_k_nearest_zipcodes_locations(query_zipcode)
            if k_nearest_zip_list and len(k_nearest_zip_list) > 0:
                # Select the similarity from similarity matrix.
                index_similarity_map = {}
                for zip_index in k_nearest_zip_list:
                    index_similarity_map[zip_index] = similarity_vector[zip_index]
            else:
                # Use all zipcodes to generate a map to hold index : similarity_value
                index_similarity_map = {index : val for index, val in enumerate(similarity_vector)} 
        else:
            # Use all zipcodes to generate a map to hold index : similarity_value
            index_similarity_map = {index : val for index, val in enumerate(similarity_vector)}
        
        # Sort the similarities in decreasing order
        sorted_similarity = [{k : index_similarity_map[k]} for k in sorted(index_similarity_map, 
                                                                           key=index_similarity_map.get, 
                                                                           reverse=True)]
        # Get top K neighbours for the given index except the index itself.
        return_list = sorted_similarity[1:k_neigh + 1]
    
        # Return result in the form of dict of index:similarity
        return_map = {}
        for index, val in enumerate(return_list):
            for k, v in val.items():
                return_map[k] = v
        # Return similarity map.
        return return_map
    
    
    def predict_rating_for_given_col(self, idx_similarity_map, col_name):
        """
        Function to predict ratings for missing value of given col_name
        in the input dataframe based on top k-similar neighbours for the
        query index.
        """
        # Get all similar zipcodes. The map holds the index of zipcodes.
        # To get the actual value, lookup those indices from zip_code list.
        similar_rows_ids = idx_similarity_map.keys()
        similar_rows_zipcode = [self._zip_code_list[k] for k in similar_rows_ids]

        # Get the rows from input dataset based on these zipcodes
        filtered_rows_df = self._ip_dataframe[self._ip_dataframe['zipcode'].isin(similar_rows_zipcode)].copy()

        # Get only desired columns from filtered rows.
        col_names = ['zipcode', col_name]
        final_df = filtered_rows_df[col_names].copy()

        # Select rows where ratings is given by neighbours for the given col_name.
        final_df = final_df[~pd.isnull(final_df[col_name])]
        codes_list = np.array(final_df['zipcode'])

        # Calculate ratings based on similarity weight.
        # predicted_rating = E(similarity_wt * value of that row,col)/total similarities.
        rating_val_list = []
        weighted_similarity_list = []
        for code in codes_list:
            rating_val = final_df[final_df['zipcode'] == code]
            rating_val = rating_val[col_name].fillna(0)
            rating_val_list.append(np.array(rating_val)[0])

            code_index = self._zip_code_list.index(code)
            weighted_similarity_list.append(idx_similarity_map[code_index])

        # Compute rating
        mul_num = np.sum(np.array(rating_val_list) * np.array(weighted_similarity_list))

        sum_den = np.sum(weighted_similarity_list)
        if mul_num == 0:
            return 0.
        return (mul_num/sum_den)
        
    
    def generate_missing_ratings_for_dataset(self, k_neigh=None, geographical_similarity=False):
        #ip_dataframe, similarity_matrix, columns_list):
        """
        Function to generate ratings values for all missing places in the given dataset.
        """
        print ("LOG: [Recommender Engine] Started calculating missing ratings.")
        num_predicted_ratings = 0
        final_predicted_ratings_records = []
        missing_ratings_map = {}

        # Get all zipcodes in the input dataframe.
        zip_list = np.array(self._ip_dataframe['zipcode']).tolist()

        # Iterate for each zipcode row and find predict missing values for ratings(if any).
        for index, code in enumerate(zip_list):
            missing_cols_list = []
            q_zipcode = code
            q_zipcode_index = index

            # Find rows similar to the zipcode.
            similar_zipcodes_map = self.get_top_n_similar_indices_and_similarities(self._similarity_matrix, 
                                                                                q_zipcode,
                                                                                q_zipcode_index,
                                                                                k_neigh=k_neigh,
                                                                                geographical_similarity=geographical_similarity)

            # Get ratings for all missing columns.
            col_ratings_list = []
            for col_name in self._non_zip_cols:    
                # Check if the value of given column for the zipcode is already present or not
                col_rating = np.array(self._ip_dataframe[self._ip_dataframe['zipcode'] == q_zipcode][col_name])[0]

                # If the rating is not given, find predicted value.
                if math.isnan(col_rating):
                    missing_cols_list.append(col_name)
                    num_predicted_ratings += 1
                    col_rating = self.predict_rating_for_given_col(similar_zipcodes_map, col_name)

                # Round off the rating
                col_rating_rounded = col_rating
                col_ratings_list.append(col_rating_rounded)

            # Add missing ratings columns for the given zipcode.
            if missing_cols_list:
                missing_ratings_map[code] = missing_cols_list

            # Add results to a tuple to form records.
            rec = (code,)
            rating_rec = tuple(col_ratings_list)

            final_rec = rec + rating_rec
            #print final_rec
            final_predicted_ratings_records.append(final_rec)

        print "Number of records: ", len(final_predicted_ratings_records)
        print "Number of missing ratings predicted: ", num_predicted_ratings
        print ("LOG: [Recommender Engine] Started calculating missing ratings.")
        return final_predicted_ratings_records, missing_ratings_map
    
    
    def get_predicted_ratings_records(self):
        """
        """
        return self._predicted_ratings_records
    
    
    def get_missing_columns_map(self):
        """
        """
        return self._missing_ratings_map
    
    def get_predicted_ratings_df(self):
        """
        """
        pred_df = generate_df_from_records(self._predicted_ratings_records, desired_columns=self._all_cols)
        return pred_df
    
        
    def perform_recommendation(self):
        """
        Step by step calling different functions to perform recommendation.
        """
        print ("LOG: [Recommender Engine] Performing recommendation.")
        # a. Convert input df to array.
        self.convert_ip_df_to_np_array()
        
        
        # b. Compute similarity matrix.
        self.generate_row_similarities_matrix(self._df_matrix_array)
        
        # c. Generate missing ratings for the input dataframe.
        self._predicted_ratings_records, self._missing_ratings_map = self.generate_missing_ratings_for_dataset(k_neigh=self._k_neigh, 
                geographical_similarity=self._geographical_similarity)
        print ("LOG: [Recommender Engine] Performed recommendation.")
