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

@app.route("/get_response", methods=['GET', 'POST'])
def get_response():
    selected_value = request.form.get('business_select').replace("_", " ")
    zipcode = request.form.get('zipcode')
    #print ("--------- Selected value:--- %s, zipcode: %s" % (selected_value, zipcode))
    
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



@app.route("/index")
def index():
    return "Welcome to Business Recommender App!"


@app.route("/business", methods=['GET', 'POST'])
def business_main_page():
    business_list = YELP_BUSINESS_TYPES_LIST
    values_list = {}
    for name in business_list:
        val = name.replace(" ", "_")
        values_list[val] = name
    print "-------------  ", values_list
    form = InputForm()
    return render_template('input.html', 
            title='Enter Information', 
            form=form,
            values_list = values_list)


@app.route("/hello/<string:name>/")
def hello(name):
    business_names = ["Restaurants", "Bars"]
    quote = business_names

    return render_template('test.html', title="Home", **locals())



@app.route("/hello/<string:name>/business.png")
def image(name):
    return "Not available"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
