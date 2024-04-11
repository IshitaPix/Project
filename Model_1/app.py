from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
df=pd.read_csv("Finbud (set 1).csv")
X = df.drop(columns=['total'])
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

#@app.route('/')
#def hello_world():
#    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

#@app.route('/', methods=['POST','GET'])


def predict():
    if request.method == 'POST':
        user_budget=float(request.form["budget"])
        input_data = [np.array(user_budget)]
        input_data = np.zeros((1, len(X.columns)))
        input_data[0, 0] = user_budget
        predicted_total_spending=model.predict(input_data)[0]
        prop = X.sum(axis=0) / X.sum().sum()
        per_category = prop* predicted_total_spending
        predicted_categories={category: budget for category, budget in zip(X.columns, per_category)}
    
   # return render_template("index.html", budget=user_budget, categories=predicted_categories)
        return render_template("result.html", categories_budget=predicted_categories, total_budget=predicted_total_spending)  #gpt
    
if __name__ == '__main__':
    app.run(debug=True)


