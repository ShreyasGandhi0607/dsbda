import pandas as pd
import nltk #natural language tool kit library widely used for NLP
# applications
import re # regular expression used for pattern matching
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Now you can use NLTK without SSL certificate verification errors

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Define the input sentence
sent = "Sachin is considered to be one of the greatest cricket players. Virat is the captain of the Indian cricket team."

# Sentence Tokenization
print("----- SENTENCES -----")
print(sent)
print("\n\n")

# Word Tokenization
print("----- WORD TOKENIZATION -----")
print(word_tokenize(sent))
print("\n\n")

# Sentence Tokenization
print("----- SENT TOKENIZATION -----")
print(sent_tokenize(sent))
print("\n\n")

# Stop Words Removal
stop_words = set(stopwords.words('english'))
token = word_tokenize(sent)
cleaned_token = [word for word in token if word.lower() not in stop_words]
words = [cleaned_word.lower() for cleaned_word in cleaned_token if cleaned_word.isalpha()]
print("----- WORDS -----")
print(words)
print("\n\n")
# STEMMING
stemmer = PorterStemmer()
port_stemmer_output = [stemmer.stem(word) for word in words]
print("----- STEMMING -----")
print(port_stemmer_output)
print("\n\n")

# LEMMATIZATION
lemmatizer = WordNetLemmatizer()
lemmatizer_output = [lemmatizer.lemmatize(word) for word in words]
print("----- LEMMATIZATION -----")
print(lemmatizer_output)
print("\n\n")

# POS TAGGING
tagged = pos_tag(cleaned_token)
print("----- POS TAGGING -----")
print(tagged)
print("\n\n")

# TF-IDF Vectorization
docs = [
    "Sachin is considered to be one of the greatest cricket players.",
    "Federer is considered one of the greatest tennis players.",
    "Nadal is considered one of the greatest tennis players.",
    "Virat is the captain of the Indian cricket team."
]

vectorizer = TfidfVectorizer(analyzer="word", norm=None, use_idf=True, smooth_idf=True)
tfidfMat = vectorizer.fit_transform(docs)
features_names = vectorizer.get_feature_names_out()
print("----- FEATURE NAMES -----")
print(features_names)
print("\n\n")

dense = tfidfMat.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=features_names)

docsList = ['Docs_1', 'Docs_2', 'Docs_3', 'Docs_4']
skDocsIfIdfdf = pd.DataFrame(tfidfMat.todense(), index=docsList, columns=features_names)
print("----- SK DOCS -----")
print(skDocsIfIdfdf)
print("\n\n")

# Cosine Similarity Calculation
csim = cosine_similarity(tfidfMat, tfidfMat)
csimDf = pd.DataFrame(csim, index=docsList, columns=docsList)
print("----- COSINE SIMILARITY -----")
print(csimDf)
print("\n\n")
