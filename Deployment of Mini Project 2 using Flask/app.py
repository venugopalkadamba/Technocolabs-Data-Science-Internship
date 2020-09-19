from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("final_model.pkl", "rb"))

def predict_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability

@app.route("/", methods = ['GET','POST'])
def home():

    education_status = ["Graduate School", "University", "High School", "Others"]
    marital_status = ["Married","Single", "Others"]

    payment_status = [
        "Account started that month with a zero balance, and never used any credit",
        "Account had a balance that was paid in full",
        "At least the minimum payment was made, but the entire balance wasn't paid",
        "Payment delay for 1 month",
        "Payment delay for 2 month",
        "Payment delay for 3 month",
        "Payment delay for 4 month",
        "Payment delay for 5 month",
        "Payment delay for 6 month",
        "Payment delay for 7 month",
        "Payment delay for 8 month",   
    ]

    alert_message = False
    success_message = False

    try:
        if request.method == 'POST':

            features = request.form.to_dict()
            features['EDUCATION'] = education_status.index(features['EDUCATION']) + 1
            features['MARRIAGE'] = marital_status.index(features['MARRIAGE']) + 1
            features['PAY_1'] = payment_status.index(features['PAY_1']) - 2

            actual_feature_names = ['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            feature_values = [features[i] for i in actual_feature_names]

            prediction, probability = predict_default(feature_values)
            if prediction[0] == 1:
                alert_message = "This account will be defaulted with a probability of {}%.".format(round(np.max(probability)*100, 2))
            else:
                success_message = "This account will not be defaulted with a probability of {}%.".format(round(np.max(probability)*100, 2))
    except:
        alert_message = "Please enter relevant information."

    return render_template("home.html", education_status = education_status, marital_status = marital_status, payment_status = payment_status, alert_message = alert_message, success_message = success_message)

if __name__ == '__main__':
    app.run(debug = True)