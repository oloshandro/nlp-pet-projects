# Corpus Processing
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


stopwords = ['a', 'about', 'an', 'am' 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

short_forms = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "it's": "it is",
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "couldn't": "could not",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    # Add more short forms and their full forms as needed
}

def replace_short_forms(text):
    # Create a regular expression pattern to match short forms as standalone words
    pattern = r'\b(?:{})\b'.format('|'.join(short_forms.keys()), re.IGNORECASE)
    
    # Replace short forms with their corresponding full forms using a lambda function
    full_forms_text = re.sub(pattern, lambda match: short_forms[match.group(0)], text)
    
    return full_forms_text


# (?) remove quotation marks, unnecessary punctuation, [{}[]\/+*%|^%#@!?()]
def punctuation_remover(text):
    pattern = r'[{}\[\]\\\/\+\*%\|\^%#@\(\)\$\"]'
    return re.sub(pattern, ' ', text)


def lemma_stopwords_token(text):
      le=WordNetLemmatizer()
      word_tokens=nltk.word_tokenize(text)
      word_tokens =[token for token in word_tokens if token.isalpha()]
      tokens=[le.lemmatize(token) for token in word_tokens if token not in stopwords and len(token)>2]
      processed_text =" ".join(tokens)
      return processed_text


# main preprocessing function
def preprocessing(text):
    reviews = replace_short_forms(text)
    reviews = punctuation_remover(reviews)
    reviews = lemma_stopwords_token(reviews)
    return reviews