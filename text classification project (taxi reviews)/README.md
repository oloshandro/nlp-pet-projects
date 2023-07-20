
## Компанія: сервіс таксі

Бізнес ціль:

Знати про що пишуть користувачі у відгуках та використовувати ці знання для покращення сервісу

На даний момент відгуки використовуються для маркетингу та для команди підтримки клієнтів, яка обробляє відгуки і визначає чи необхідно швидко на них реагувати

**Завдання:**

Запропонуйте свої ідеї як можна використати відгуки для досягнення результату.

**Інформація, яка може бути корисною:**

 ⁃ Визначення емоційного забарвлення відгуку (позитивний / негативний контекст) *sentiment analyses*;
 
 ⁃ **Топік (тематика)**, до якої відноситься відгук *topic modelling*;
 
 ⁃ Саммері по кожній темі *priority*

*Визначення топіку та/або важливих атрибутів допомагає в створенні системи пріоритезації відгуків і швикого реагування на найбільш важливі
до пріоритетних можуть відноситись відгуки про загублені речі, про небезпечне водіння, про зухвале ставлення водія, несправність автомобіля тощо

# Implementation

## Pre-annotation steps for analysing the data set using unsupersised ML approach

**1. Topic modelling** 
*to define  possible topics to use for annotation*

Gensim vs. Scikit-learn (LDA: Latent Dirichlet Allocation - topic modeling technique to extract topics from a given corpus)

BERTopic


**2. Text Clustering**
*to sort the reviews into possible categories*

**K-Means** is one of the most popular "clustering" algorithms. K-means stores  $k$  centroids that it uses to define clusters. A point is considered to be in a particular cluster if it is closer to that cluster's centroid than any other centroid.

K-Means finds the best centroids by alternating between

(1) assigning data points to clusters based on the current centroids (2) chosing centroids (points which are the center of a cluster) based on the current assignment of data points to clusters.


**3.Sentiment analysis**
*to run sentiment analysis predictions* Used default **transformers** model **distilbert-base-uncased-finetuned-sst-2-english** and **sentiment-roberta-large-english**(which showed better result). 


## Post annotation steps

**1. Data cleaning & preprocessing:**
* handling Contractions and negations: don't/isn't... -> do not, is not
* removing irrelevant punctuation r'[{}\[\]\\\/\+\*%\|\^%#@\(\)\$\"]' | spaces | numbers? |
* removing stop words
* lowercase
* tokenization or sentence splitting
* choose data

**2. Feature engineering**
* Tf-idf

**3. Training sentiment analysis model**
* using MultinomialNB, LinearSVC, LogisticRegression, AdaBoostClassifier to evaluate  and compare the prediction accuracy (*accuracy, precision, recall, and F1-score* metrix)

**4. Fine-tuning the model**
*by adding SMOTE Technique to handle imbalanced dataset*

## Conclusion1:
MultinomialNB with SMOTE technique gave the best results so far

**5. Training topic classification model**