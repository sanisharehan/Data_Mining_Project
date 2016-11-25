'''
data_preprocessing_engine.py: This file holds class with various methods used for
preprocessing Yelp business review data.

Author: Sanisha Rehan
'''
from constants import YELP_BUSINESS_TYPES_LIST, YELP_FILTERED_STRING_COLUMNS_LIST
from utility import convert_string_from_unicode_to_ascii


class Data_Preprocessing_Engine(object):
    """
    Class for performing business/project specific pre-processing on
    cleaned yelp business data.
    """
    def __init__(self, input_dataframe):
        self._ip_dataframe = input_dataframe
        self._business_type_cat_map = dict()
        self._unmatched_categories = []
        self.perform_data_preprocessing()
    
        
    def is_business_category_mapped(self, cat_name):
        """
        Function to chcek if a business category is already mapped
        to some business type.
        """
        for val in self._business_type_cat_map.values():
            for category in val:
                if category == cat_name:
                    return True
        return False
        
        
    def generate_business_types_categories_map(self):
        """
        Function to generate a map associating different business categories
        available in Yelp data with important business types.
        """
        print ("LOG: [Preprocessing Engine] Generating business types map.")
        business_map = {}
        for k in YELP_BUSINESS_TYPES_LIST:
            business_map[k] = []
        
        # Add food as one of the categories for restaurants.
        business_map['Restaurants'] = ['Food']
        
        # Iterate over all the rows of dataframe to create a map between business 
        # keywords and important business types.
        for index, row in self._ip_dataframe.iterrows():
            category_list = row['categories']

            has_important_category = False
            buisness_name = None

            # Associate food related places/businesses with restaurants.
            for category in category_list:
                if category == 'Food':
                    has_important_category = True
                    buisness_name = 'Restaurants'
                    break
                    
                if category in YELP_BUSINESS_TYPES_LIST:
                    has_important_category = True
                    buisness_name = category
                    break

            # Map the categories to business types.
            if has_important_category:
                for k in category_list:
                    if (k != buisness_name) and (k not in business_map[buisness_name]) \
                            and (k not in YELP_BUSINESS_TYPES_LIST) and (k != 'Food'):
                        is_already_mapped = self.is_business_category_mapped(k)
                        if not is_already_mapped:
                            business_map[buisness_name].append(k)
            else:
                self._unmatched_categories.append(row)
        
        self._business_type_cat_map = business_map
        #print self._business_type_cat_map
        print ("LOG: [Preprocessing Engine] Generated business types map.")

        
    def get_business_types_categories_map(self):
        """
        """
        return self._business_type_cat_map
    
    
    def fill_business_types_info_in_df(self):
        """
        Function to fill the column business_type in df according to the types associated with
        different business categories.
        """
        print ("LOG: [Preprocessing Engine] Filling business types.")
        filled_df = self._ip_dataframe.copy()      
        for index, row in filled_df.iterrows():
            categories_list = row['categories']
            
            if len(categories_list) == 0:
                continue
            
            business_type = None
            
            # Check if a business type is present in categories
            # e.g. Restaurant, Pets etc.
            common = set(categories_list) & set(YELP_BUSINESS_TYPES_LIST)
            if len(common) != 0:
                business_type = common.pop()
                filled_df.set_value(index, 'business_type', business_type)
                continue

            # Else try to find the related business_type.
            cat_found = False
            for key, val in self._business_type_cat_map.iteritems():
                common = set(categories_list) & set(val)
                if len(common) != 0:
                    cat_found = True
                    business_type = key
                    break

            # Inplace setting the column value in dataframe.
            filled_df.set_value(index, 'business_type', business_type)
        filled_df = filled_df[filled_df['business_type'] != '']
        
        self._ip_dataframe = filled_df.copy()
        # Select only those where business_type != ''
        # self._ip_dataframe = self._ip_dataframe[self._ip_dataframe['business_type'] != '']
        #print self._ip_dataframe[self._ip_dataframe['business_type'] == '']
        print ("LOG: [Preprocessing Engine] Filled business types.")
        
            
    def convert_columns_to_ascii_df(self):
        """
        Function to convert all columns in dataframe to ascii.
        """
        print ("LOG: [Preprocessing Engine] Generating preprocessed dataframe.")
        for col_name in YELP_FILTERED_STRING_COLUMNS_LIST:
            self._ip_dataframe[col_name] = self._ip_dataframe[col_name].apply(convert_string_from_unicode_to_ascii)
            
        print ("LOG: [Preprocessing Engine] Generated preprocessed dataframe.")
         
            
    def get_final_preprocessed_df(self):
        """
        """
        print ("\tNumber of rows in final preprocessed dataframe: %d" % len(self._ip_dataframe.index))
        return self._ip_dataframe

    
    def perform_data_preprocessing(self):
        """
        Step by step function calling to generate the final preprocessed Yelp df.
        """
        print ("LOG: [Preprocessing Engine] Starting data preprocessing.")
        self.generate_business_types_categories_map()
        self.fill_business_types_info_in_df()
        self.convert_columns_to_ascii_df()
        print ("LOG: [Preprocessing Engine] Finished data preprocessing.")
             
