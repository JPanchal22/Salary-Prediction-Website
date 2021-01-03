import numpy as np 
import pickle 
from flask import Flask, render_template, request, redirect, url_for 

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def prediction():
    name = request.form["name"]
    
    if name.strip(" ") == "":
        return render_template("index.html", pred="Please enter your name.")
    else:
        name = name.strip(" ").title()
        experience = request.form["exp"]
        if experience.strip(" ")  == "": 
            return render_template("index.html", pred="Please enter your experience.")
        elif experience.strip(" ").isnumeric() == False:
            return render_template("index.html", pred="Your experience should be an integer value.") 
        else:  
            experience = int(experience.strip(" "))
            final_X = np.array(experience).reshape(-1,1)
            prediction = np.array(model.predict(final_X))
            output = "{0:.3f}".format(prediction[0][0])

            return render_template("index.html", pred=f"{name}, your predicted annual salary is Rs. {output}")
       
if __name__ == "__main__":
    app.run(debug=True)