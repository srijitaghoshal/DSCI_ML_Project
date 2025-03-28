from flask import Flask, render_template, request, render_template_string
import os
import openai 
from dotenv import load_dotenv, find_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())  # This loads variables from .env file into environment

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a list to store conversation context
messages = []  # This will hold the conversation context for ChatGPT

# Ticker List for User Context
# PLACEHOLDER FOR NOW
ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'V', 'BABA']


# Neural net recommended ticker_list
neural_net_tickers = []

# graph_suggestions = True

# Time frame for investment length
timeframe = 7 # SET TO A WEEK FOR NOW

# Transaction type
buy = 1

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
    messages[0]['content'] += 'The response should be no more than 3 sentences.'
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

#Function to get array of tickers with the highest predicted percent change   
def get_top_tickers():
    prompt = '''Pretend you are a character in a movie that is a stock market expert who is able to provide 
        real-time financial predictions and recommendations. Based on current events and news, 
        predict which 25 S&P 500 companies will have the largest percent change (positive or negative) in price in the next month. 
        Output the ticker symbols in a python string. Only output the string of tickers with no variable name.'''
    
    try:
        response = openai.chat.completions.create(
            model= 'gpt-4o-mini',
            messages=[{"role": "user",'content':prompt}],
            temperature=0.0
        )
        response = response.choices[0].message.content
        tickers = [ticker.strip('"') for ticker in response.split(',')]
        return tickers
    except Exception as e:
        return f"Error: {str(e)}"

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    chat_response = None  # Default response if none is received
    ticker_string = "No suggestions generated yet"
    suggest_for_me = False  
    # graphs = []
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        transaction_type = request.form.get('transaction')
        if transaction_type == 'buy':
            buy = True
        else:
            buy = False

        if user_input.lower() in "suggest for me":
            ticker_list = get_top_tickers()
            neural_net_tickers = predict_and_decide(ticker_list, timeframe, buy).tolist()
            ticker_string = " ".join(str(x) for x in neural_net_tickers)
            print(ticker_string)
            chat_response = f"Here is our tuned neural net generated suggestion: {ticker_string}"
            add_to_context(chat_response, "system")

            # print(neural_net_tickers)
            # for ticker in neural_net_tickers:
            #     data = import_data(ticker, timeframe)
            #     if data is not None:
            #         graph_html = print_candles(data)
            #         graphs.append({"ticker": ticker, "graph_html": graph_html})

        # if 'chat' in request.form:  # 'chat' is the name of the button
        #     if request.form['chat'] == 'submit':

        # if request.form['action'] == 'submit':
        if chat_response is None:
            risk_tolerance = request.form.get('risk_tolerance', 5)  # Default to 5 if not provided
            length_of_investment = request.form.get('length')
            # transaction_type = request.form.get('transaction')
            # Add user input and risk tolerance to context (if necessary)
            if user_input:
                add_to_context(f"Risk Tolerance: {risk_tolerance}, Seeking length of investment: {length_of_investment}, User choice to buy or sell: {transaction_type}", "system")  
                add_to_context(user_input, "user")

                # Get response from OpenAI's GPT model based on the conversation context
                chat_response = get_completion_from_messages(messages, model="gpt-4")

                # Add the model's response to the context
                add_to_context(chat_response, "assistant")

    return render_template('index.html', chat_response=chat_response)

# Route to the second page (second_page.html)
@app.route('/graph', methods=['GET', 'POST'])
def graphing():
    graph_html = None
    user_input = request.form.get('user_input')
    if user_input:
        data = import_data(user_input, timeframe)
        if data is not None:
            graph_html = print_candles(data, user_input)
    return render_template('graph.html', graph=graph_html)  # Renders the second page

# Plotting function for candlestick patterns
def print_candles(df, ticker):
    # Create a figure with two rows for subplots
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.3, 
                        row_heights=[0.7])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close']),
                  row=1, col=1)
    
    fig.update_layout(width=800, height=800, title=ticker)
    return fig.to_html(full_html=False)

def import_data(stock, timeframe):
    data = yf.download(stock, period="2y", interval='1d')
    data.columns = data.columns.get_level_values(0)
    data['EMA_50'] = ta.ema(data['Close'], length=50)
    data['EMA_200'] = ta.ema(data['Close'], length=200)
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['Pct_Change'] = data['Close'].pct_change(periods=timeframe).shift(-timeframe)  # Percent change over X days
    data.dropna(inplace=True)  # Remove any rows with NaN values
    return data

def prepare_lstm_data(data, feature_columns, target_column, time_steps=60):
    X = []
    Y = []
    for i in range(time_steps, len(data) - 5):  # Ensure we have future data for Y
        X.append(data[feature_columns].iloc[i - time_steps:i].values)
        Y.append(data[target_column].iloc[i])
    return np.array(X), np.array(Y)

def build_training_data(tickers, timeframe):
    # sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # sp500_table = pd.read_html(sp500_url)
    # TICKERS = sp500_table[0]['Symbol'].tolist()

    # # Select a subset for faster testing
    # TICKERS = TICKERS[:10]  # Adjust as needed

    # Prepare data for training
    feature_columns = ['Close', 'EMA_50', 'EMA_200', 'SMA_50', 'SMA_200', 'RSI']
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    time_steps = 60  # Number of time steps for LSTM
    scaler = MinMaxScaler()

    for ticker in tickers:
        # print(f"Processing {ticker}...")
        try:
            data = import_data(ticker, timeframe)
            data[feature_columns] = scaler.fit_transform(data[feature_columns])
            X, Y = prepare_lstm_data(data, feature_columns, 'Pct_Change', time_steps)
            if len(X) > 0 and len(Y) > 0:
                # length = len(X)
                # index = 0.8*length
                # X_train.extend(X[:index])
                # Y_train.extend(Y[:index])
                # X_test.extend(X[index:])
                # Y_test.extend(Y[index:])
                X_train.extend(X)
                Y_train.extend(Y)

        except Exception as e:
            print(f"Skipping {ticker} due to an error: {e}")

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

def build_and_train_model(X_train, y_train):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        LSTM(32),
        Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse')
    # TODO: Change epochs to 20 to optimize for testing, originally 50
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)
    return model

def predict_and_decide(tickers, tf, buy):
    # Ensure tickers is a numpy array
    tickers = np.array(tickers)
    timeframe = tf
    
    # Prepare training data
    X_train, y_train = build_training_data(tickers, timeframe)
    
    # Check if training data is prepared
    if len(X_train) == 0 or len(y_train) == 0:
        print("Training data is empty. Ensure the tickers array is valid and data was processed correctly.")
        return np.array([])

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Initialize a list to store the results
    feature_columns = ['Close', 'EMA_50', 'EMA_200', 'SMA_50', 'SMA_200', 'RSI']
    time_steps = 60
    scaler = MinMaxScaler()
    predictions = []

    for ticker in tickers:
        try:
            # Download the latest data for the ticker
            data = import_data(ticker, timeframe)
            
            # Scale feature columns
            data[feature_columns] = scaler.fit_transform(data[feature_columns])
            
            # Extract the last `time_steps` rows for prediction
            if len(data) >= time_steps:
                last_data = data[feature_columns].iloc[-time_steps:].values
                last_data = np.expand_dims(last_data, axis=0)  # Reshape for LSTM input
                
                # Make a prediction
                predicted_change = model.predict(last_data)[0][0]
                predictions.append((ticker, predicted_change))
            else:
                print(f"Not enough data for {ticker} to make a prediction.")
        except Exception as e:
            print(f"Skipping {ticker} due to an error: {e}")

    # Sort predictions by predicted price change in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Extract tickers and predictions into separate arrays
    sorted_tickers = np.array([item[0] for item in predictions])
    y_pred = np.array([item[1] for item in predictions])
    
    # Generate buy or sell signals based on the buy flag
    if buy:
        selected_tickers = sorted_tickers[y_pred > 0]  # Buy if predicted change > 0
    else:
        selected_tickers = sorted_tickers[y_pred <= 0]  # Sell if predicted change <= 0

    return selected_tickers
    
# Run the Flask app
if __name__ == '__main__':

    app.run(debug=True)
