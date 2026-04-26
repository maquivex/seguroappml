from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('./model/insurance-ml.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    age = None

    if request.method == 'POST':
        age = int(request.form['age'])

        age_sc = sc_x.transform(np.array([[age]]))
        pred = model.predict(age_sc)
        prediction = sc_y.inverse_transform(pred)[0][0]

    return render_template('index.html', prediction=prediction, age=age)

if __name__ == '__main__':
    app.run(debug=True)