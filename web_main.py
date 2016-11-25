"""
Simple Flask application for providing a simple UI for the recommender
application.
"""
from flask import Flask, flash, redirect, render_template, request, session
from flask_wtf import Form
from wtforms import StringField, BooleanField
from main_app import get_ratings_for_business_zipcode
from constants import YELP_BUSINESS_TYPES_LIST

app = Flask(__name__)
app.config.from_object('config')

class InputForm(Form):
    business_id = StringField('business_id')
    zipcode = StringField('zipcode')

# Displaying response code.
@app.route("/business/response", methods=['GET', 'POST'])
def get_response():
    """
    This URL gets called when displaying response for the user input.
    It gets the predicted rating from the recommendation main application
    and displays it on UI.
    """
    selected_value = request.form.get('business_select').replace("_", " ")
    zipcode = request.form.get('zipcode')
    result_map = {}
    result_map['business'] = selected_value
    result_map['zipcode'] = zipcode
    print ("--------- Selected value:--- %s, zipcode: %s" % (result_map['business'], zipcode))
    rating = get_ratings_for_business_zipcode(selected_value, int(zipcode))
    
    if rating is None:
        rating = "Oops! Bad idea to start this business in the zipcode! Not sufficient data to predict rating!"
    result_map['rating'] = rating
    return render_template('response.html',
            title='Response',
            result_map=result_map)

# Simple URL for testing the app.
@app.route("/index")
def index():
    return "Welcome to Business Recommender App!"

# Home page for application.
@app.route("/business/home_page", methods=['GET', 'POST'])
def business_main_page():
    """
    Method for displaying application home page. It gets all the available business
    types an ddisplays in the dropdown.
    """
    business_list = YELP_BUSINESS_TYPES_LIST
    values_list = {}
    for name in business_list:
        val = name.replace(" ", "_")
        values_list[val] = name
    form = InputForm()
    return render_template('input.html', 
            title='Enter Information', 
            form=form,
            values_list = values_list)

# Run the application.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
