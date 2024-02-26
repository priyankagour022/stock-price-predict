from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load the model
model = load_model('stock_prediction_model.pkl')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the dataset
df = pd.read_csv('CSIOY.csv')
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * .8))
test_data = dataset[training_data_len - 60:, :]

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    days = int(request.json['days'])

    # Create the x_test dataset
    x_test = []
    for i in range(60, len(test_data) + days):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the predicted prices for the next 'days' days
    predicted_prices = predictions[-days:]

    return jsonify({'predicted_prices': predicted_prices.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
