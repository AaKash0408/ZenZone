import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import json

with open('intents.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])
df

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)
df

df['tag'].unique()

import plotly.graph_objects as go

intent_counts = df['tag'].value_counts()
#fig = go.Figure(data=[go.Bar(x=intent_counts.index, y=intent_counts.values)])
#fig.update_layout(title='Distribution of Intents', xaxis_title='Intents', yaxis_title='Count')
#fig.show()

df['pattern_count'] = df['patterns'].apply(lambda x: len(x))
df['response_count'] = df['responses'].apply(lambda x: len(x))
avg_pattern_count = df.groupby('tag')['pattern_count'].mean()
avg_response_count = df.groupby('tag')['response_count'].mean()

#fig = go.Figure()
#fig.add_trace(go.Bar(x=avg_pattern_count.index, y=avg_pattern_count.values, name='Average Pattern Count'))
#fig.add_trace(go.Bar(x=avg_response_count.index, y=avg_response_count.values, name='Average Response Count'))
#fig.update_layout(title='Pattern and Response Analysis', xaxis_title='Intents', yaxis_title='Average Count')
#fig.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import plotly.graph_objects as go
# Split the dataset into training and testing sets
X = df['patterns']
y = df['tag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = SVC()
model.fit(X_train_vec, y_train)

# Predict intents for the testing set
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Convert float values in the report to dictionaries
report = {label: {metric: report[label][metric] for metric in report[label]} for label in report if isinstance(report[label], dict)}

# Extract evaluation metrics
labels = list(report.keys())
evaluation_metrics = ['precision', 'recall', 'f1-score']
metric_scores = {metric: [report[label][metric] for label in labels if label in report] for metric in evaluation_metrics}

#fig = go.Figure()
#for metric in evaluation_metrics:
   # fig.add_trace(go.Bar(name=metric, x=labels, y=metric_scores[metric]))

#fig.update_layout(title='Intent Prediction Model Performance',
                  #xaxis_title='Intent',
                  #yaxis_title='Score',
                  #barmode='group')

#fig.show()

def predict_intent(user_input):
    # Vectorize the user input
    user_input_vec = vectorizer.transform([user_input])

    # Predict the intent
    intent = model.predict(user_input_vec)[0]
    
    return intent

# Function to generate responses based on predicted intents
def generate_response(intent):
    # Implement your logic here to generate appropriate responses based on the predicted intents
    if intent == 'greeting':
        response = "Hello! How can I assist you today?"
    elif intent == 'goodbye':
        response = "Goodbye! Take care."
    elif intent == 'question':
        response = "I'm sorry, I don't have the information you're looking for."
    elif intent == 'afternoon':
        response = "Good afternoon. How is your day going?"
    elif intent == 'about':
        response = "I'm Euphoria, your Personal Therapeutic AI Assistant. How are you feeling today?" 
    elif intent == 'creation':
        response = "I was created by Team Celestial. I was trained on a text dataset using Deep Learning & Natural Language Processing techniques "
    elif intent == 'name':
        response = "Oh nice to meet you. Tell me how was your week?"
    elif intent == 'sad':
        response = "I'm sorry to hear that. I'm here for you. Talking about it might help. So, tell me why do you think you're feeling this way?"
    elif intent == 'stressed':
        response = "Take a deep breath and gather your thoughts. Go take a walk if possible. Stay hydrated."
    elif intent == 'worthless':
        response = "It's only natural to feel this way. Tell me more. What else is on your mind?"
    elif intent == 'depressed':
        response = "Sometimes when we are depressed, it is hard to care about anything. It can be hard to do the simplest of things. Give yourself time to heal."
    elif intent == 'happy':
        response = "That's great to hear. I'm glad you're feeling this way."
    elif intent=='casual':
        response="I'm listening. Please go on."
    elif intent=='not-talking':
        response="Talking about something really helps. If you're not ready to open up then that's ok. Just know that I'm here for you."
    elif intent=='scared':
        response="It'll all be okay. This feeling is only momentary."
    elif intent=='death':
        response="My condolences. I'm here if you need to talk."
    elif intent=='understand':
        response="I'm trying my best to help you. So please talk to me."
    elif intent=='done':
        response="Oh okay we're done for today then. See you later"
    elif intent=='hate-me':
        response="I'm sorry if i have exhibited any sort of behaviour to make you think that."
    else:
        response = "I'm here to help. Please let me know how I can assist you."
    return response

if __name__ == "__main__":
    # Get user input from command line arguments
    user_input = sys.argv[1]
    
    # Predict intent
    intent = predict_intent(user_input)

    # Generate response
    response = generate_response(intent)
    
    # Output the response (this will be captured by Node.js)
    print(response)
