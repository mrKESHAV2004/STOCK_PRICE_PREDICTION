# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

# Global variables to store the model and metrics
model = None
mse = None
r2 = None
X_test = None
y_test = None
y_pred = None

def initialize_model():
    global model, mse, r2, X_test, y_test, y_pred
    
    try:
        # Loading the dataset
        data = pd.read_csv('stock_price_data.csv')
        
        # Features and target
        X = data[['Low', 'High']]
        y = data['Next_Day_Price']
        
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Making predictions
        y_pred = model.predict(X_test)
        
        # Calculating metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return True
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return False

def create_plots():
    # Scatter plot
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', size=8, opacity=0.7)
    ))
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    scatter.update_layout(
        title='Predicted vs Actual Prices',
        xaxis_title='Actual Prices',
        yaxis_title='Predicted Prices',
        template='plotly_white'
    )
    
    # Trend plot
    trend = go.Figure()
    trend.add_trace(go.Scatter(
        y=y_test.values,
        mode='lines+markers',
        name='Actual Prices',
        line=dict(color='green')
    ))
    trend.add_trace(go.Scatter(
        y=y_pred,
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='blue', dash='dash')
    ))
    
    trend.update_layout(
        title='Trend of Predicted vs Actual Prices',
        xaxis_title='Index',
        yaxis_title='Stock Price',
        template='plotly_white'
    )
    
    return json.dumps(scatter, cls=plotly.utils.PlotlyJSONEncoder), json.dumps(trend, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def home():
    # Initialize model if not already initialized
    if model is None:
        success = initialize_model()
        if not success:
            return "Error: Failed to initialize the model. Please check your data file and try again."
    
    # Create plots
    try:
        scatter_json, trend_json = create_plots()
        
        return render_template('index.html',
                             mse=f"{mse:.2f}",
                             r2=f"{r2:.2f}",
                             scatter_json=scatter_json,
                             trend_json=trend_json)
    except Exception as e:
        return f"Error creating plots: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is initialized
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized'
            })
        
        # Get values from the form
        low = float(request.form['low'])
        high = float(request.form['high'])
        
        # Make prediction
        new_data = pd.DataFrame([[low, high]], columns=['Low', 'High'])
        prediction = model.predict(new_data)[0]
        
        return jsonify({
            'success': True,
            'prediction': f"{prediction:.2f}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Initialize the model before starting the app
    initialize_model()
    app.run(debug=True)