import tkinter as tk
import requests
from flask import Flask, request, jsonify
import threading
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample chatbot intents
intents = [
    {"intent": "greetings", "patterns": ["Hello", "Hi", "Hey"], "responses": ["Hello! How can I assist you?"]},
    {"intent": "farewell", "patterns": ["Bye", "Goodbye"], "responses": ["Goodbye! Have a great day!"]},
    {"intent": "institute", "patterns": ["Tell me about your institute", "Where are you located?"], "responses": ["We are Social Prachar Institute."]}
]

# Prepare training data for Naïve Bayes model
training_sentences, training_labels = [], []
words, class_names = [], []

for intent in intents:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        training_sentences.append(pattern.lower())
        training_labels.append(intent['intent'])

        if intent['intent'] not in class_names:
            class_names.append(intent['intent'])

words = sorted(set(words))

# Vectorizing text input
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences).toarray()
# Train Naïve Bayes model
y = np.array([class_names.index(label) for label in training_labels])
model = GaussianNB()
model.fit(X, y)

# Flask API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    processed_input = vectorizer.transform([user_message.lower()]).toarray()
    prediction = model.predict(processed_input)
    intent = class_names[prediction[0]]

    for item in intents:
        if item["intent"] == intent:
            return jsonify({"response": random.choice(item["responses"])})
    
    return jsonify({"response": "I'm sorry, I don't understand."})

# Run Flask in a separate thread
def run_flask():
    app.run(port=7500, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Tkinter GUI
def send_message(event=None):
    user_message = user_input.get()
    if user_message.strip():
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"😺: {user_message}\n", "user")

        response = requests.post("http://127.0.0.1:7000/chat", json={"message": user_message}).json()["response"]
        chat_window.insert(tk.END, f"🤖: {response}\n\n", "bot")

        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)
        user_input.delete(0, tk.END)

# Set up Tkinter window
root = tk.Tk()
root.title("AI Chatbot")

chat_window = tk.Text(root, height=20, width=50, state=tk.DISABLED, wrap=tk.WORD)
chat_window.tag_configure("user", justify="right", foreground="blue")
chat_window.tag_configure("bot", justify="left", foreground="green")
chat_window.pack(pady=10)

user_input = tk.Entry(root, width=50)
user_input.pack(pady=10)
user_input.bind("<Return>", send_message)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

root.mainloop()
