# Multi-Output Time-Series Forecasting: Stock Market Prediction

**Author:** Yash Gautam  

## 📌 Project Overview
This project implements a multi-output **Long Short-Term Memory (LSTM)** neural network to forecast stock market trends. Unlike standard univariate time-series models that predict a single future value, this architecture is designed to handle multiple input features and predict multiple steps into the future simultaneously.

Specifically, the model utilizes a sliding window approach to take the **last 10 days** of historical stock data across **11 distinct financial features** to predict the values of all 11 features for the **next 5 consecutive days**.

## 📊 Dataset
The model is trained and evaluated on the **NIFTY-50 Stock Market Data (2000 - 2021)**, with a specific focus on the `SHREECEM` (Shree Cement) stock dataset. 

* **Source:** [Kaggle: NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
* **Time Span:** January 1, 2000, to April 30, 2021.
* **Features Tracked (11):** `Prev Close`, `Open`, `High`, `Low`, `Last`, `Close`, `VWAP`, `Volume`, `Turnover`, `Deliverable Volume`, `%Deliverble`.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras (LSTM, Dense, Reshape layers)
* **Data Manipulation:** Pandas, NumPy
* **Preprocessing & Metrics:** Scikit-learn (`MinMaxScaler`, MSE, MAE)
* **Data Visualization:** Matplotlib

## ⚙️ Model Architecture & Methodology
1. **Data Preprocessing:** 
   * Forward and backward fill applied to handle any missing data points.
   * `MinMaxScaler` applied to normalize all 11 features between a range of 0 and 1, ensuring stable gradient descent during neural network training.
2. **Sequence Generation:** 
   * Transformed the time-series data into supervised learning sequences.
   * **Input Shape:** `(Samples, 10, 11)`
   * **Output Shape:** `(Samples, 5, 11)`
3. **Network Design:**
   * **LSTM Layer:** 64 units with ReLU activation to capture temporal dependencies.
   * **Dense Hidden Layer:** 128 units with ReLU activation for feature extraction.
   * **Dense Output Layer:** 55 units (representing 5 days × 11 features).
   * **Reshape Layer:** Formats the final output back to a `(5, 11)` dimensional array.
4. **Optimization:** Compiled using the `Adam` optimizer and optimized against Mean Squared Error (`mse`).

## 📈 Evaluation & Results
The model is evaluated on a strict chronological train-test split (80% / 20%) to prevent data leakage. Predictions are inverse-transformed back to their original scale before calculating final metrics.

**Performance Metrics Tracked:**
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

The project also includes automated visualization scripts to plot the **Date-wise Predictions vs. True Known Values** (specifically targeting the `Close` price) to provide an intuitive visual assessment of the model's directional accuracy over the forecast horizon.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
