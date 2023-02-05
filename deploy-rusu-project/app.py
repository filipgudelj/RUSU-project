from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('prediction.html')

@app.route("/prediction", methods=['POST'])
def prediction():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    model = str(request.form['model'])
    lr_prediction = lr_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    rf_prediction = rf_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    knn_prediction = knn_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    lr_output = lr_prediction[0]
    rf_output = rf_prediction[0]
    knn_output = knn_prediction[0]

    if (model == 'lr'):
        if (lr_output == 0):
            return render_template('prediction.html', prediction_result = "The person doesn't have a heart disease")
        else: 
            return render_template('prediction.html', prediction_result = "The person has a heart disease")
    elif (model == 'rf'):
        if (rf_output == 0):
            return render_template('prediction.html', prediction_result = "The person doesn't have a heart disease")
        else: 
            return render_template('prediction.html', prediction_result = "The person has a heart disease")
    else:
        if (knn_output == 0):
            return render_template('prediction.html', prediction_result = "The person doesn't have a heart disease")
        else: 
            return render_template('prediction.html', prediction_result = "The person has a heart disease")

if __name__ == "__main__":
    app.run()