# Receipt Count Prediction for Fetch

This project is focused on predicting the number of scanned receipts for Fetch on a monthly basis using machine learning (ML). The aim is to develop an algorithm that can predict the approximate number of receipts for each month of 2022 based on daily data from 2021.

## Table of Contents

- [Data](#data)
- [Solution Overview](#solution-overview)
- [App Overview](#app-overview)
- [Dockerization](#dockerization)
- [Setup Instructions](#setup-instructions)
  - [Running Locally](#running-locally)
  - [Running via Docker](#running-via-docker)
- [Model Details](#model-details)
- [Further Improvements](#further-improvements)


## Data

The dataset contains daily counts of scanned receipts for the entire year of 2021. The model is trained on this data to make monthly predictions for 2022. The columns in the dataset are:

- `# Date`: The date of the receipts (format: YYYY-MM-DD)
- `Receipt_Count`: The number of receipts scanned on that date

## Solution Overview

This project is built using PyTorch for the machine learning model and Flask for serving a web-based interface where users can upload CSV files, visualize the actual and predicted receipt counts, and see the monthly predictions for 2022.

The project includes:
- A Long Short-Term Memory (LSTM) model implemented in PyTorch for time-series prediction.
- A web app built using Flask that allows users to upload receipt data and view predictions for future months in a graphical format.
- The application is containerized using Docker for easy deployment and usage.

## App Overview

The Flask app includes:
- A form to upload daily receipt data (CSV format).
- Visualization of both the actual and predicted receipt counts.
- Monthly receipt predictions for 2022 displayed in a list format.

## Dockerization

The app is packaged using Docker, allowing for easy setup and deployment. The `Dockerfile` ensures that all dependencies are installed and exposes the necessary port to access the Flask app.

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install torch pandas numpy matplotlib flask

EXPOSE 5000

CMD ["python", "app.py"]
```

## Setup Instructions

### Running Locally

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies**:
   Ensure you have Python 3.9 installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:
   ```bash
   python app.py
   ```

4. **Access the app**:
   Once the app is running, open your browser and go to `http://localhost:5000`. You can upload the CSV file containing the daily receipt counts and view the predictions.

### Running via Docker

To run the app inside a Docker container, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t receipt-prediction .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 5000:5000 receipt-prediction
   ```

3. **Access the app**:
   Open your browser and go to `http://localhost:5000`. You can now upload the data and view predictions.

## Model Details

The model used for the receipt prediction task is a Long Short-Term Memory (LSTM) neural network implemented from scratch using PyTorch. The LSTM is trained on the normalized daily receipt counts and predicts the monthly aggregated receipt counts for the next 12 months.

### Training

The model is trained on the 2021 daily receipt data. Here's the process:
- Data is normalized to ensure the model training is stable.
- A sequence of the last 3 days is used to predict the next month's aggregated count.
- The model is optimized using Mean Squared Error (MSE) loss and the Adam optimizer.

### Inference

Once the model is trained, it is used in the Flask app to predict the receipt counts for each month of 2022 based on the input data. The predictions are denormalized before being displayed.

## Further Improvements

Future improvements to this project could include:
- Tuning the hyperparameters of the LSTM model to improve prediction accuracy.
- Adding more complex features (e.g., external data) to enhance the model's performance.
- Improving the web interface to provide a more interactive experience for the user.
- Packaging the trained model and app into a REST API for broader integration.

