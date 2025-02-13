import tkinter as tk
from tkinter import scrolledtext
import pickle
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Define the intents list
intents = [
    # Define your intents as before
    {
        "intent": "greetings",
        "patterns": ["Hello", "Hi", "Good day", "Hey", "How are you?", "Good morning", "Hi there"],
        "responses": [
            "Hello! Welcome to Social Prachar Institute. How can I assist you today?",
            "Hi there! How can I help you with your queries about our courses?",
            "Good day! I’m Vajra.AI, your guide to the Social Prachar Institute. What would you like to know?"
        ]
    },
    {
        "intent": "farewells",
        "patterns": ["Goodbye", "Bye", "Thank you", "See you later", "Take care", "Farewell"],
        "responses": [
            "Thank you for visiting! Have a great day!",
            "It was my pleasure to assist you. Take care and good luck with your learning journey!",
            "Goodbye, and feel free to come back if you have more questions!"
        ]
    },
    {
        "intent": "institute_details",
        "patterns": ["What is the name of the institute?", "Where are you located?", "Tell me about your institute?",
                     "What is Social Prachar Institute?"],
        "responses": [
            "The name of the institute is Social Prachar.",
            "Social Prachar is an educational institute offering courses in Data Science & AI, Data Analytics, Python Full Stack, Java Full Stack, and AWS Developer. We focus on providing high-quality training to our students with a strong emphasis on hands-on experience and practical knowledge.",
            "We are located at [hi-tech city road, 216, Manjeera Majestic Commercial, JNTU Rd, near JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY, Hyderabad, Telangana 500085]."
        ]
    },
    {
        "intent": "course_info",
        "patterns": ["What courses do you offer?", "Tell me about the courses", "What courses are available?",
                     "Can you list your courses?"],
        "responses": [
            "We offer Data Science, Data Analytics, Python Full Stack, Java Full Stack, and AWS Developer courses.",
            "Our courses include Data Science & AI, Data Analytics, Python Full Stack, Java Full Stack, and AWS Developer. You can learn more about each course on our website."
        ]
    },
    {
        "intent": "fee_structure",
        "patterns": ["How much does the course cost?", "What are the fees for the courses?",
                     "Tell me the course fees."],
        "responses": [
            "The fee for Data Science is 50k, and other courses are 30k.",
            "Our courses are priced as follows: Data Science - 50k, Data Analytics - 30k, Python Full Stack - 30k, Java Full Stack - 30k, AWS Developer - 30k."
        ]
    },
    {
        "intent": "internship_opportunity",
        "patterns": ["Do you offer internships?", "Are there internship opportunities?",
                     "Can I get an internship after completing the course?"],
        "responses": [
            "Yes, internships are available for some courses like Data Science and Python Full Stack.",
            "We offer internships for selected courses such as Data Science and Python Full Stack. Internships are based on performance and course requirements."
        ]
    },
    {
        "intent": "placement_assistance",
        "patterns": ["Do you provide placement assistance?", "Will you help me get a job?",
                     "Can you help me with jobs?"],
        "responses": [
            "Yes, we provide placement assistance, including resume building and interview preparation.",
            "We offer placement assistance services such as resume building, mock interviews, job placement, and referrals."
        ]
    },
    {
        "intent": "refund_policy",
        "patterns": ["What is your refund policy?", "Can I get a refund if I don't like the course?",
                     "Do you have a refund policy for your courses?"],
        "responses": [
            "We offer a 7-day refund policy if you're not satisfied with the course.",
            "If you're not satisfied with the course, we offer a 7-day refund policy."
        ]
    },
    {
        "intent": "demo_classes",
        "patterns": ["Do you offer demo classes?", "Can I try a demo class?", "Where can I book a demo class?"],
        "responses": [
            "You can book your demo class by visiting [Social Prachar Institute](https://socialprachar.com/). Demo classes are available for all courses.",
            "Demo classes are available for all courses. You can book your demo class on our website."
        ]
    },
    {
        "intent": "payment_methods",
        "patterns": ["What are the payment options?", "What payment methods do you accept?",
                     "Can I pay online for the courses?"],
        "responses": [
            "We accept all types of payments.",
            "You can pay via credit card, debit card, online banking, or other accepted methods. All payment options are listed on our website."
        ]
    },
    {
        "intent": "error_handling",
        "patterns": ["I didn't understand.", "I don't know what you mean.", "Can you explain it again?"],
        "responses": [
            "Sorry, I didn’t quite get that. Could you please rephrase your question?",
            "I didn't quite understand. Could you please rephrase it?"
        ]
    },
    {
        "intent": "spelling_mistake",
        "patterns": ["I think I made a spelling mistake.", "I might have typed it wrong.",
                     "Could you check for spelling mistakes?"],
        "responses": [
            "I noticed a spelling mistake in your question. Could you please check and rephrase it?",
            "There seems to be a spelling mistake. Could you please check and correct it?"
        ]
    },
    {
        "intent": "institute_name",
        "patterns": ["What is the name of the institute?", "Which institute is this?", "Tell me the institute name",
                     "What is the name of your institute?"],
        "responses": [
            "The name of the institute is Social Prachar."
        ]
    }
]

# Initialize Lemmatizer and other settings
lemmatizer = nltk.WordNetLemmatizer()
ignore_words = ['?', '!', '.', ',']

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Function to preprocess and lemmatize user input
def preprocess_sentence(sentence):
    sentence_tokens = nltk.word_tokenize(sentence.lower())  # Tokenize
    sentence_tokens = [lemmatizer.lemmatize(word) for word in sentence_tokens if word not in ignore_words]
    return " ".join(sentence_tokens)

# Preparing training data
training_sentences = []
training_labels = []

for intent in intents:
    for pattern in intent['patterns']:
        processed_sentence = preprocess_sentence(pattern)  # Apply preprocessing
        training_sentences.append(processed_sentence)
        training_labels.append(intent['intent'])

# Train the vectorizer
X = vectorizer.fit_transform(training_sentences).toarray()

# Encode the labels
y = training_labels

# Save the trained vectorizer
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# Load the saved model (Naive Bayes model)
with open('model.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)

# Function to predict intent
def predict_intent(user_input):
    # Preprocess user input
    user_input_tokens = preprocess_sentence(user_input)

    # Convert the user input into the same format as the training data (using the vectorizer)
    input_vector = vectorizer.transform([user_input_tokens])

    # Convert to a DataFrame to retain feature names
    input_vector_df = pd.DataFrame(input_vector.toarray(), columns=vectorizer.get_feature_names_out())

    # Predict the intent using the loaded model
    prediction = nb_model.predict(input_vector_df)[0]

    # Based on the predicted intent, generate a response
    for intent in intents:
        if intent['intent'] == prediction:
            return random.choice(intent['responses'])

    return "Sorry, I didn’t understand that. Could you please rephrase?"

# Tkinter GUI Setup
window = tk.Tk()
window.title("Vajra.AI Chatbot")
window.geometry("400x500")

# ScrolledText widget to display chat history
chatbot_output = scrolledtext.ScrolledText(window, height=20, width=40, font=("Arial", 12))
chatbot_output.pack(pady=10)
chatbot_output.config(state=tk.DISABLED)

# Entry widget for user input
user_entry = tk.Entry(window, font=("Arial", 14), width=35)
user_entry.pack(pady=10)

# Function to handle button click and get response
def on_send_button_click():
    user_input = user_entry.get()
    if user_input.lower() == 'quit':
        window.quit()
        return
    response = predict_intent(user_input)
    chatbot_output.config(state=tk.NORMAL)
    chatbot_output.insert(tk.END, f"You: {user_input}\nVajra.AI: {response}\n\n")
    chatbot_output.config(state=tk.DISABLED)
    user_entry.delete(0, tk.END)

# Button to send input
send_button = tk.Button(window, text="Send", font=("Arial", 14), command=on_send_button_click)
send_button.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()
