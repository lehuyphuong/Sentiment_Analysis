from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import contractions
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('./IMDB-Dataset.csv')

# remove duplicated rows
df = df.drop_duplicates()

# preprocess data
stop = set(stopwords.words('english'))

# expanding contractions
WORDS_LEN = "words length"


def expand_contractions(text):
    return contractions.fix(text)


def preprocess_text(text):
    wl = WordNetLemmatizer()

    soup = BeautifulSoup(text, 'html.parser')  # remove html tag

    text = soup.get_text()
    text = expand_contractions(text)

    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)

    text = emoji_clean.sub(r'', text)
    text = re.sub(r'\.(?=\S)', '. ', text)  # add space after full stop
    text = re.sub(r'http\S+', '', text)  # remove urls
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])  # remove punctuation and make text lower case
    text = " ".join([
        wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()
    ])
    return text


df['review'] = df['review'].apply(preprocess_text)
# create autocpt arguments


def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)


freq_pos = len(df[df['sentiment'] == 'positive'])
freq_neg = len(df[df['sentiment'] == 'negative'])

data = [freq_pos, freq_neg]

labels = ['positive', 'negative']
# create a pie chart
pie, ax = plt.subplots(figsize=[11, 7])
plt.pie(x=data, autopct=lambda pct: func(pct, data), explode=[0.0025]*2,
        pctdistance=0.5, colors=[sns.color_palette()[0], 'tab:red'],
        textprops={'fontsize': 16})

labels = [r'Positive', r'Negative']
plt.legend(labels, loc="best", prop={'size': 14})
# pie.savefig("piechart.png")

word_lens = df['review'].str.split().map(lambda x: len(x))
df_temp = df.copy()
df_temp['words length'] = word_lens

hist_positive = sns.displot(
    data=df_temp[df_temp['sentiment'] == 'positive'],
    x=WORDS_LEN, hue="sentiment", kde=True, height=7,
    aspect=1.1, legend=False
).set(title="Words in positive reviews")

hist_negative = sns.displot(
    data=df_temp[df_temp['sentiment'] == 'negative'],
    x=WORDS_LEN, hue="sentiment", kde=True, height=7,
    aspect=1.1, legend=False, palette=['red']
).set(title="Words in positive reviews")

plt.figure(figsize=(7, 7.1))
kernel_distribution_number_words_plot = sns.kdeplot(
    data=df_temp, x=WORDS_LEN, hue="sentiment", fill=True,
    palette=[sns.color_palette()[0], 'red']
).set(title="words in reviews")
plt.legend(title="sentiment", labels=['negative', 'positive'])
# plt.show()

label_encode = LabelEncoder()
y_data = label_encode.fit_transform(df['sentiment'])


# split training dataset
x_train, x_test, y_train, y_test = train_test_split(
    df['review'], y_data, test_size=0.2, random_state=42
)

tfidf_vectorizer = TfidfVectorizer(max_features=10000)
tfidf_vectorizer.fit(df['review'], df['sentiment'])

x_train_encoded = tfidf_vectorizer.transform(x_train)
x_test_encoded = tfidf_vectorizer.transform(x_test)

# call model
dt_classifier = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42,
    ccp_alpha=0.0,
)

# train
dt_classifier.fit(x_train_encoded, y_train)

# evaluate
y_pred = dt_classifier.predict(x_test_encoded)
print("accuracy for use Decision Tree: ", accuracy_score(y_pred, y_test))

# call model
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    ccp_alpha=0.0,
    random_state=42,
)

# train
rf_classifier.fit(x_train_encoded, y_train)

# evaluate
y_pred = rf_classifier.predict(x_test_encoded)

print("accuracy for use Random forest: ", accuracy_score(y_pred, y_test))
