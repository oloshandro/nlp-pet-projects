from flask import Flask, render_template, request
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv()

app = Flask(__name__)

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content


def get_classification(review):
    classification_result = []
    
    sentiment_result = get_completion(prompt_sentiment)
    classification_result.append(sentiment_result)
    if "negative" in sentiment_result.lower():
        topic_result = get_completion(prompt_topic)
        identified_topics = [topic for topic in topics if topic.lower() in topic_result.lower()]
        classification_result.append(topic_result)
        if "1 Pricing and Fairness" or "3 Driver behaviour" or "6 Lost things" or "8 Safety & reliability" in identified_topics:
            alert = "ALERT: high priority!"
            classification_result.append(alert)
    else:
        summary = get_completion(prompt_summary)
        classification_result.append(summary)
    
    print("Classification Result:", classification_result)
    return classification_result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        results = get_classification(review)
        return render_template('index.html', review=review, results=results)
    else:
        return render_template('index.html', review='', results=[])


if __name__ == "__main__":
    
    # review = input("Please, type your review: ")
    topics = ['1 Pricing and Fairness', '2 Driver professionalism', '3 Driver behaviour', '4 Customer Service', '5 Application', '6 Lost things', '7 Vehicle Condition', '8 Safety & reliability', '9 Generally bad']

    prompt_sentiment = f"""
    Define the sentiment of the following taxi service review, \
    which is delimited with triple backticks.
    Give your answer in such format: "Sentiment: positive" or "Sentiment: negative"

    """ 

    prompt_summary = f"""
    Your task is to extract relevant information from a taxi service review \
    to give summary to the Marketing department highlighting the benefits of the service. 
    Limit your summary to 20 words.

    
    """

    prompt_topic = f"""
    Define which of the  '''{topics}''' are mentioned in the review.
    Here is the description for each topic:
    1 Pricing and Fairness: fare structure, pricing transparency, fairness in charging, affordability, hidden costs, overcharged, wrongly charged, payment methods, cost and refund issues;
    2 Driver professionalism: driver's performance and professionalism, punctuality, navigation skills, driver's cancellations, late pick-up & drop-off at the wrong place, address-related problems, driver didn't deliver to the door;
    3 Driver behaviour: russian language/music, communication, and friendliness, racist/aggressive behavious, demanding something, driver cheated / stole something;
    4 Customer Service: taxi company, it's responsiveness to queries or complaints, helpfulness of customer support;
    5 Application: ease and efficiency of the booking process, user-friendliness of the booking platform or app, app problems;
    6 Lost things: effectiveness in returning the things left in the vehicle,
    not delivered order;
    7 Vehicle Condition: cleanliness, maintenance, smell, smoke-free environment, safety & reliability, place for luggage, comfort, cold;
    8 Safety & reliability: violation of traffic rules, talking on phone, safety belt, reckless/fast driving, road accident, drink driving
    9 Generally bad: bad experience - nothing from the above mentioned,
    Give your answer in a dictionary format: "Topic" : " ... "
    State all the topics mentioned in the review. If so, write  "Topics" as the key, and all the topics list as the values.

    
    """

    # results = get_classification(review)
    # print(results)

    app.run(debug=True)