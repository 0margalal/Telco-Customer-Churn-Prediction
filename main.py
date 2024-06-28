import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("flask")
install("joblib")
install("scikit-learn")

import pandas as pd
import flask
from flask import render_template
from flask import request

categoryColumn = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                  'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

numericalColumn = ['tenure', 'MonthlyCharges', 'TotalCharges']

categoricalFeatures = {}

df_ = pd.read_csv(r'D:\UNI\YEAR 3\sem 2\Data Viz\Project\data\Dataset2.csv')
df_.dropna(inplace=True)
for column_ in categoryColumn:
    categoricalFeatures[column_] = df_[column_].unique().tolist()


def preprocessData(data):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder()
    scaler = StandardScaler()
    data['SeniorCitizen'] = data['SeniorCitizen'].astype('object')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(inplace=True)
    data['SeniorCitizen'] = le.fit_transform(data['SeniorCitizen'])
    for column_ in categoryColumn:
        data[column_] = le.fit_transform(data[column_])
    data['customerID'] = le.fit_transform(data['customerID'])
    for column_ in numericalColumn:
        data[column_] = scaler.fit_transform(data[column_].values.reshape(-1, 1))
    data.drop('TotalCharges', axis=1, inplace=True)
    data = data.reindex(sorted(data.columns), axis=1)
    return data


def predictChurn(data, MLModel):
    data = preprocessData(data)
    return MLModel.predict(data)[0]


def getListOfModels():
    import os
    models = []
    base_path = os.path.dirname(__file__)
    os.chdir(os.path.join(base_path, 'models'))
    for model in os.listdir():
        models.append(model)
    return models


app = flask.Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', models=getListOfModels(), categoricalFeatures=categoricalFeatures)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    row = {}
    print(data.keys())
    for key in data.keys():
        if key in categoryColumn or key in numericalColumn:
            row[key] = [data[key]]
    row['customerID'] = "7590-VHVEG"
    df = pd.DataFrame(row)
    import os
    base_dir = os.path.dirname(__file__)
    model = os.path.join(base_dir, 'models', data['model'])
    import joblib
    MLModel = joblib.load(model)
    prediction = predictChurn(df, MLModel)
    return render_template('index.html', prediction=prediction, models=getListOfModels(), categoricalFeatures=categoricalFeatures)


@app.route('/train', methods=['POST'])
def train():
    import os
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'Dataset2.csv')
    import joblib
    from sklearn.linear_model import LogisticRegression
    df = pd.read_csv(data_path)
    df = preprocessData(df)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = LogisticRegression()
    model.fit(X, y)
    model_path = os.path.join(base_dir, 'models')
    import random
    joblib.dump(model, os.path.join(model_path, "logisticRegression" + str(random.randint(1, 1000)) + ".pkl"))
    acc = model.score(X, y)
    return render_template('index.html', models=getListOfModels(), categoricalFeatures=categoricalFeatures, acc=acc)


app.run(port=5001)
