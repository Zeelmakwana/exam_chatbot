from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import random
import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Load the necessary data and model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'E:\ddect chatbot\intents.json').read())
words = pickle.load(open(r'words.pkl', 'rb'))
classes = pickle.load(open(r'classes.pkl', 'rb'))
model = load_model(r'chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)

ERROR_THRESHOLD = 0.25
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    print("Predicted Probabilities:", res)  # Add this line for debugging
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print("Filtered Results:", results)  # Add this line for debugging
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        # Sort the intents based on probability and select the one with the highest probability
        intents_list.sort(key=lambda x: float(x['probability']), reverse=True)
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                # Return the response with the highest probability
                return i['responses'][0]
    else:
        return "I'm sorry, I didn't understand that."

def create_tables():
    db.create_all()
    
@app.route("/")
def home():
    return render_template("register.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='sha256')

        new_user = User(username=username, email=email, phone=phone, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Successfully logged in
            return render_template('index.html')  # Redirect to the 'index.html' page (or your desired page)

        # Incorrect username or password
        return "Invalid username or password"

    return render_template('login.html')

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_message = request.form["user_message"]
    ints = predict_class(user_message)
    res = get_response(ints, intents)
    return res  

if __name__ == "__main__":
    with app.app_context():
        create_tables()  # Create the database tables before running the app
        app.run(debug=True)
