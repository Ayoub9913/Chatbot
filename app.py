from flask import Flask, request, render_template
from chatbot import chatbot_response  # Import your chatbot function
from keras.models import load_model  # Import Keras model loading

app = Flask(__name__)

# Load your chatbot model
chatbot_model = load_model('chatbot_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    chatbot_output = chatbot_response(user_input)  # Pass the loaded model
    #return render_template('index.html', chatbot_response=chatbot_output)
    return chatbot_output

if __name__ == '__main__':
    app.run(debug=True)
