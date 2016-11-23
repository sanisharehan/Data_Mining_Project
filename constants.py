'''
constants.py: File holding all constant values needed for the project.

Author: Sanisha Rehan
'''
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

