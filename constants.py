'''
constants.py: File holding all constant values needed for the project.

Author: Sanisha Rehan
'''
# Constants for arguments.
VALID_RUN_MODES = ['all', 'train', 'test']
VALID_SIMILARITY_TYPES = ['cosine', 'geo']
VALID_TEST_RUN_MODES = ['fast', 'all']

# Filenames for saving different files.
FILTERED_FILENAME = "yelp_filtered_data_stage_1.csv"
PREPROCESSED_FILENAME = "yelp_preprocessed_data_stage_2.csv"
PIVOT_FILENAME = "yelp_pivot_data_stage_3.csv"
TRAINING_FILENAME = "yelp_training_data_actual.csv"
TEST_FILENAME = "yelp_test_data_actual.csv"
TRAINING_PRED_FILENAME = "yelp_training_data_predicted.csv"
TEST_PRED_FILENAME = "yelp_test_data_predicted.csv"


# Define all constants here.
YELP_BUSINESS_DATA_FILENAME = 'yelp_academic_dataset_business.json'

YELP_DESIRED_COLUMNS_LIST = ['categories', 
                            'business_type', 
                            'state', 
                            'city', 
                            'full_address', 
                            'zipcode', 
                            'longitude', 
                            'latitude', 
                            'stars_rating', 
                            'type', 
                            'review_count'
                       ]
YELP_BUSINESS_TYPES_LIST = ['Restaurants',
                            'Shopping',
                            'Bars',
                            'Nightlife',
                            'Active Life',
                            'Health & Medical',
                            'Home Services',
                            'Beauty & Spas',
                            'Event Planning & Services',
                            'Arts & Entertainment',
                            'Local Services',
                            'Fashion',
                            'Automotive',
                            'Professional Services',
                            'Doctors',
                            'Hotels & Travel',
                            'Education',
                            'Financial Services',
                            'Real Estate',
                            'Religious Organizations',
                            'Public Services & Government',
                            'Pets'
                           ]
YELP_FILTERED_STRING_COLUMNS_LIST = ['full_address',
                                    'city',
                                    'state',
                                    'type']

