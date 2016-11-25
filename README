###############################################################################

CMPE239: Web and Data Mining Project
====================================
Team 4
======
Business Recommendation Engine
==============================

The application has been designed and implemented as part of group project for
Spring 16 session.

Brief Introduction: This recommendation system is built to predict ratings for a 
given business type and zipcode using Yelp Business reviews data.

How To Run application:
======================

I. Run the UI application:
python web_main.py


II. Run the recommendation engine on terminal (using training and test data):

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

Note: Follow the logs to view data saved from different processing and recommendation
stages.

