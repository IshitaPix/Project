from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)

# Load the trained model
model2 = pickle.load(open('model2.pkl', 'rb'))

# Load the CSV file into a DataFrame
df = pd.read_csv("Finbud (set 1).csv")

# Get the list of features
features = df.columns.tolist()
features.remove("total")  # Remove the target column from features
categories = features.copy()
@app.route('/')
def index():
    return render_template('index2.html',categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        budget = float(request.form['budget'])
        top_features = [request.form['first'], request.form['second'], request.form['third']]
        remaining_features = [feature for feature in features if feature not in top_features]

        num_remaining_features = len(remaining_features)
        allocated_budget = {}
        remaining_budget = budget

        for feature in top_features:
            allocated_amount = remaining_budget / (num_remaining_features + len(top_features))
            allocated_budget[feature] = allocated_amount
            remaining_budget -= allocated_amount

        remaining_feature_means = df[remaining_features].mean()
        remaining_total_mean = df[remaining_features].mean().sum()
        for feature in remaining_features:
            feature_mean = df[feature].mean()
            allocated_budget[feature] = budget * (feature_mean / remaining_total_mean)

        total_allocated_budget = sum(allocated_budget.values())
        allocation_factor = budget / total_allocated_budget
        for feature in allocated_budget:
            allocated_budget[feature] *= allocation_factor

        budget_allocation = [allocated_budget[feature] for category in categories]
        input_data = np.array(budget_allocation).reshape(1, -1)
        predicted_total_budget = model2.predict(input_data)[0]

        #return render_template('result2.html', allocated_budget=allocated_budget, predicted_total_budget=predicted_total_budget)
        return render_template('result2.html', allocated_budget=allocated_budget)

if __name__ == '__main__':
    app.run(debug=True)
