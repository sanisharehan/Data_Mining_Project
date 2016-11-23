'''
data_pivoting_engine.py: Class used for pivoting the Yelp business review data.

Author: Sanisha Rehan
'''
from utility import round_to_closest_pt_5

class Data_Pivoting_Aggregation_Engine(object):
    """
    Class for performing data pivoting, aggreagation of ratings on preprocessed
    data.
    """
    def __init__(self, ip_dataframe):
        self._ip_dataframe = ip_dataframe
        self._agg_ratings_df = None
        self._pivot_df = None
        
        self._group_by_cols_list = ['business_type', 'zipcode']
        self._agg_cols_list = ['review_count', 'all_stars']
        self._pivot_cols_to_sel_list = ['business_type', 'zipcode', 'agg_review_stars_sum']
        self.perform_data_aggregation_and_pivoting()
        
    
    def add_aggregated_ratings_col(self):
        """
        Function to add aggregated review ratings column in the dataframe.
        """
        print ("LOG: [Pivoting Engine] Started aggregated ratings.")
        self._ip_dataframe['all_stars'] = self._ip_dataframe['stars_rating'] * self._ip_dataframe['review_count']
        print ("LOG: [Pivoting Engine] Finished aggregated ratings.")
        
    
    def compute_aggregated_ratings_by_cols(self, group_by_cols_list, agg_cols_list):
        """
        Function to compute aggregated ratings from the given ratings
        based on group by cols passed.
        aggregated ratings = Sum of ratings/total review counts
        """
        print ("LOG: [Pivoting Engine] Started computing aggregated ratings.")
        aggregations = {k : 'sum' for k in agg_cols_list}
        self._agg_ratings_df = self._ip_dataframe.groupby(group_by_cols_list).agg(aggregations)
        
        self._agg_ratings_df['agg_review_stars'] = self._agg_ratings_df['all_stars']/self._agg_ratings_df['review_count']
        # Round the values to closest 0.5
        self._agg_ratings_df['agg_review_stars'] = self._agg_ratings_df['agg_review_stars'].apply(round_to_closest_pt_5)
        
        # This will generate columns with names: business_type, zipcode
        # review_count_sum, all_stars_sum, agg_review_stars_sum
        self._agg_ratings_df = self._agg_ratings_df.add_suffix('_sum').reset_index()
        #print self._agg_ratings_df[:12]
        print ("LOG: [Pivoting Engine] Finished computing aggregated ratings.")
        
        
    def pivot_aggregated_table_by_column(self, index_col, col_to_pivot, col_to_fill, cols_to_select_list):
        """
        Function selects columns specified by cols_to_select_list from the aggregated
        dataframe and creates a pivoted table based on col_to_pivot value.
        """
        print ("LOG: [Pivoting Engine] Started pivoting.")
        # Select only required columns from the df.
        filtered_df = self._agg_ratings_df[cols_to_select_list]
        
        # To pivot the dataframe w.r.t. business_type column
        # Hint: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pivot.html
        self._pivot_df = filtered_df.pivot(index= index_col, 
                                                 columns= col_to_pivot, 
                                                 values= col_to_fill)
        #print self._pivot_df[:12]
        print ("LOG: [Pivoting Engine] Finished pivoting.")
        
        
    def perform_data_aggregation_and_pivoting(self):
        """
        """
        print ("LOG: [Pivoting Engine] Started pivoting and aggregation.")
        self.add_aggregated_ratings_col()
        self.compute_aggregated_ratings_by_cols(self._group_by_cols_list, self._agg_cols_list)
        self.pivot_aggregated_table_by_column(index_col = 'zipcode', 
                                              col_to_pivot = 'business_type', 
                                              col_to_fill = 'agg_review_stars_sum',
                                              cols_to_select_list = self._pivot_cols_to_sel_list)
        print ("LOG: [Pivoting Engine] Finished pivoting and aggregation.")
        
    
    def get_pivoted_df(self):
        """
        """
        print ("LOG: [Pivoting Engine] Number of rows in pivoted dataframe: %d" % len(self._pivot_df.index))
        pivot_df = self._pivot_df.reset_index()
        #pivot_df = pivot_df[YELP_BUSINESS_TYPES_LIST]
        return pivot_df
        
