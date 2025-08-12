import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import urllib.request


print("Reading text and tokenizing...")

response = urllib.request.urlopen('https://www.gutenberg.org/files/42324/42324-h/42324-h.htm')

html = response.read()
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(strip=True)

s_tokens = sent_tokenize(text)
w_tokens = word_tokenize(text)

# removing punctuation
print("Removing Punctuation...")
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)
tokens = [t.lower() for t in tokens]

print("Removing Stop Words...")
clean_tokens = tokens[:]
sr = stopwords.words("english")
for token in tokens:
    if token in stopwords.words("english"):
        clean_tokens.remove(token)

print("Stemming...")
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in clean_tokens]

print("Lemmatizing...")
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]

# obtaining word counts
freq = nltk.FreqDist(lemmatized_tokens) # lowercase, non-punctuated 
for key,val in freq.items():
    print(str(key) + ":" + str(val))
print("Length of Unique Items:", len(freq.items()))
# freq.plot(20, cumulative=False)
# plt.show()


print("POS Analysis")
import operator
pos = nltk.pos_tag(lemmatized_tokens)
pos_counts = {}
for key, val in pos:
    print(str(key) + ":" + str(val))
    if val not in pos_counts.keys():
        pos_counts[val] = 1
    else:
        pos_counts[val] += 1

print(pos_counts)
plt.bar(range(len(pos_counts)), list(pos_counts.values()), align="center")
plt.xticks(range(len(pos_counts)), list(pos_counts.keys()))
# plt.show()


print("Tri-Grams...")
from nltk import ngrams
trigrams = ngrams(text.split(), 3)
for gram in trigrams: print(gram)

print("Document-Term Matrix")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

response = urllib.request.urlopen("https://www.gutenberg.org/cache/epub/18247/pg18247-images.html")
html = response.read()
soup = BeautifulSoup(html, "html.parser")
text2 = soup.get_text(strip=True)

docs = [text, text2]
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

print("Instances of 'fear' in both documents:")
print(df["fear"]) # shows the count for this word in both documents

print("Instances of 'hope' in both documents:")
print(df["hope"]) # shows the count for this word in both documents

print(df)