from flask import Flask, render_template, request
import os
import openai 
from dotenv import load_dotenv, find_dotenv

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())  # This loads variables from .env file into environment

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a list to store conversation context
messages = []  # This will hold the conversation context for ChatGPT

# Function to add messages to the context
def add_to_context(message, role):
    """
    Add a message to the conversation context.
    Args:
        message (str): The content of the message.
        role (str): The role of the sender ('user', 'assistant', or 'system').
    """
    if role not in ['user', 'assistant', 'system']:
        raise ValueError("Role must be 'user', 'assistant', or 'system'.")
    
    messages.append({'role': role, 'content': message})

# Function to get a response from OpenAI's GPT model
def get_completion_from_messages(messages, model="gpt-4", temperature=0.7):
    """
    Get a completion (response) from OpenAI's GPT model based on the conversation context.
    Args:
        messages (list): List of message dictionaries to be sent to the model.
        model (str): The model to use, such as "gpt-3.5-turbo".
        temperature (float): Controls randomness in the output.
    Returns:
        str: The response content from the model.
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    chat_response = None  # Default response if none is received
    if request.method == 'POST':
        # Get the user input and slider value
        user_input = request.form.get('user_input')
        risk_tolerance = request.form.get('risk_tolerance', 5)  # Default to 5 if not provided

        # Add user input and risk tolerance to context (if necessary)
        if user_input:
            add_to_context(f"Risk Tolerance: {risk_tolerance}", "system")  # Add risk tolerance to context (optional)
            add_to_context(user_input, "user")

            # Get response from OpenAI's GPT model based on the conversation context
            chat_response = get_completion_from_messages(messages, model="gpt-4")

            # Add the model's response to the context
            add_to_context(chat_response, "assistant")

    return render_template('index.html', chat_response=chat_response)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def print_candles(df):
    # Create a figure with two rows for subplots
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.3, 
                        row_heights=[0.7],
                        subplot_titles=("Candlestick Chart"))

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close']),
                  row=1, col=1)
    
    fig.update_layout(width=1200, height=800)
    fig.show()

import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def import_data(stock, startDate):
    data = yf.download(stock, start=startDate, interval='1d')
    data.columns = data.columns.get_level_values(0)
    data['EMA_50'] = ta.ema(data['Close'], length=50)
    data['EMA_200'] = ta.ema(data['Close'], length=200)
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['Pct_Change_5D'] = data['Close'].pct_change(periods=5).shift(-5)  # Percent change over 5 days
    data.dropna(inplace=True)  # Remove any rows with NaN values
    return data

def prepare_lstm_data(data, feature_columns, target_column, time_steps=60):
    X = []
    Y = []
    for i in range(time_steps, len(data) - 5):  # Ensure we have future data for Y
        X.append(data[feature_columns].iloc[i - time_steps:i].values)
        Y.append(data[target_column].iloc[i])
    return np.array(X), np.array(Y)

def build_training_data():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(sp500_url)
    TICKERS = sp500_table[0]['Symbol'].tolist()

    # Select a subset for faster testing
    TICKERS = TICKERS[:10]  # Adjust as needed

    # Prepare data for training
    feature_columns = ['Close', 'EMA_50', 'EMA_200', 'SMA_50', 'SMA_200', 'RSI']
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    time_steps = 60  # Number of time steps for LSTM
    scaler = MinMaxScaler()

    for ticker in TICKERS:
        # print(f"Processing {ticker}...")
        try:
            data = import_data(ticker, startDate='2023-01-01')
            data[feature_columns] = scaler.fit_transform(data[feature_columns])
            X, Y = prepare_lstm_data(data, feature_columns, 'Pct_Change_5D', time_steps)
            if len(X) > 0 and len(Y) > 0:
                length = len(X)
                index = 0.8*length
                X_train.extend(X[:index])
                Y_train.extend(Y[:index])
                X_test.extend(X[index:])
                Y_test.extend(Y[index:])

        except Exception as e:
            print(f"Skipping {ticker} due to an error: {e}")

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

def build_and_train_model(X_train, y_train):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        LSTM(32),
        Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
    return model

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
