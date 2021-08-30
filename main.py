from flask import Flask, render_template, request
from werkzeug.utils import redirect
import price

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def hello():
    global ans
    ans = 0
    if request.method == "POST":
        lat = request.form['lat']
        lon = request.form['lon']
        age = request.form['age']
        rooms = request.form['rooms']
        bedr = request.form['bedr']
        pop = request.form['pop']
        house = request.form['house']
        median_income = request.form['median_income']
        op_ocean = request.form['op_ocean']
        op_inland = request.form['op_inland']
        op_island = request.form['op_island']
        op_bay = request.form['op_bay']
        op_near_ocean = request.form['op_near_ocean']
        ans = price.price_prediction(abc=[lat, lon, age, rooms, bedr, pop, house,
                                          median_income, op_ocean, op_inland, op_island, op_bay, op_near_ocean])

    return render_template("index.html", pred=ans)


"""
@app.route("/submit",methods=['POST'])
def submit():
    #HTML -> .py
    if request.method == "POST":
        name = request.form["username"]

    #.py -> HTML
    return render_template("submit.html",n = name)
"""

if __name__ == '__main__':
    app.run(debug=True, port=3333)
