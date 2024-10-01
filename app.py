from flask import Flask, request, render_template
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
from model import LSTMModel
import torch
import numpy as np

app = Flask(__name__)

# Helper function to create sequences 
def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        xs.append(x)
    return np.array(xs)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        print("POST request received")
        if 'datafile' not in request.files:
            print("No file part in the request")
            return "No file uploaded", 400  # Handle missing file
        
        file = request.files['datafile']
        if file.filename == '':
            print("No file selected")
            return "No file selected", 400  # Handle no file selected

        print("File uploaded: ", file.filename)

        # Handle file upload and prediction
        file = request.files['datafile']
        data = pd.read_csv(file, parse_dates=['# Date'])

        # Group and prepare the data (assuming you want monthly receipt counts)
        data['Month'] = data['# Date'].dt.to_period('M')
        monthly_data = data.groupby('Month')['Receipt_Count'].sum().reset_index()
        monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
        monthly_counts = monthly_data['Receipt_Count'].values.astype(float)

        # Normalize the data
        data_min = monthly_counts.min()
        data_max = monthly_counts.max()
        monthly_counts_norm = (monthly_counts - data_min) / (data_max - data_min)
        print(monthly_counts_norm)
        # Load the model
        model = LSTMModel()
        model.load_state_dict(torch.load('lstm_model.pth'))  # Load your pre-trained model
        model.eval()

        # Prepare the input sequence
        seq_length = 3  # Make sure this matches the training sequence length
        input_seq = monthly_counts_norm[-seq_length:].tolist()

        # Predict the next 12 months
        future_predictions = []
        for _ in range(12):
            seq_input = torch.tensor(input_seq[-seq_length:]).float().unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                pred = model(seq_input)
                future_predictions.append(pred.item())
                input_seq.append(pred.item())

        # Denormalize the predictions
        future_predictions_denorm = [(pred * (data_max - data_min)) + data_min for pred in future_predictions]
        last_month = monthly_data['Month'].iloc[-1]
        future_dates = pd.date_range(last_month + pd.DateOffset(months=1), periods=12, freq='M')

        # Plotting the actual and predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(monthly_data['Month'], monthly_counts, label='Actual')
        plt.plot(future_dates, future_predictions_denorm, label='Predicted', linestyle='--')
        plt.title('Receipt Counts')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Close the plot to free memory
        plt.close()

        # Prepare predictions for display in the template
        predictions = [(date.strftime("%Y-%m"), int(pred)) for date, pred in zip(future_dates, future_predictions_denorm)]
        print("predictions: ", predictions)
        return render_template('index.html', plot_url=plot_url, predictions=predictions)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
