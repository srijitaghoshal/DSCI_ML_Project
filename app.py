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

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
