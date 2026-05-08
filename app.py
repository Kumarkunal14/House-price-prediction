from flask import Flask, request, render_template
import pickle
import json
import numpy as np

# Load model
with open('banglore_home_prices_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Load columns
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    locations = data_columns[3:]
    return render_template('index.html', locations=locations)


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Create input array
    x = np.zeros(len(data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Set location column = 1
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # Predict
    prediction = model.predict([x])[0]

    output = round(prediction, 2)

    return render_template(
        'index.html',
        prediction_text=f"Estimated Price: ₹ {output} Lakhs",
        locations=data_columns[3:]
    )


if __name__ == "__main__":
    app.run(debug=True)