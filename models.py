#Library for Data Preprocessing Analysing Visualizing
import pandas as pd
#Library for Text Data Preprocessing 
import nltk
import re
import string
# Library for Splitting Data into Training and Testing
from sklearn.model_selection import train_test_split
# Library for converting text into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# Library for Machine Learning Models/ Estimators
# Logisitic Regression
from sklearn.linear_model import LogisticRegression
# Support Vector Machine
from sklearn import svm
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# Library for Machine Learning Models/ Estimators Evaluation Pattern
from sklearn.metrics import classification_report, confusion_matrix
# Model Save and Load 
import pickle

data = pd.read_csv("Tweets_Mix.csv")

data.head(10)

data.info()

data['Sentiment'].value_counts()

# Data Preprocessing
# Writing Function to remove the mentions  URL's  and String with @

def removeURL(text):
    tweet_out = re.sub(r'@[A-Za-z0-9]+','',text)
    re.sub('https?://[A-zA-z0-9]+','',text)
    return tweet_out

# Writing function to remove the non-numeric characters
def removeNonAlphanumeric(text):
    text_out = "".join([char for char in text if char not in string.punctuation])
    return text_out

data["Tweet_No_URL"]  = data["Tweets"].apply(lambda x:removeURL(x))
data["Tweet_No_Punc"] = data["Tweet_No_URL"].apply(lambda x:removeNonAlphanumeric(x))

data.head()

#Tokenization

def tokenization(text):
    token = re.split('\W+',text)
    return token

data ["Tokens"] = data["Tweet_No_Punc"].apply(lambda x:tokenization(x))

data.head()

#Stemming

ps = nltk.PorterStemmer()

def stemming (text):
    out_text = [ps.stem(word) for word in text]
    return out_text

data['Stem'] = data['Tokens'].apply(lambda x:stemming(x))

data.head()

#Lemmatizing

wn = nltk.WordNetLemmatizer()

def lemmatize(text):
    out_text = [wn.lemmatize(word) for word in text]
    return out_text

data['Lem'] =data['Tokens'].apply(lambda x:lemmatize(x))

data['Lem'].head()

data.head()

#Stop Words

stopwords = nltk.corpus.stopwords.words('english')

def remove_stopWords(token_list):
    text_out = [word for word in token_list if word not in stopwords]
    return text_out

data['StopRemove'] = data['Lem'].apply(lambda x:remove_stopWords(x))

data.head()

def final_join(token):
    document = " ".join([word for word in token if not word.isdigit()])
    return document

data['FinalJoin'] = data['StopRemove'].apply(lambda x:final_join(x))

data.head()

# Model Building

# Splitting of Data Set


X = data['FinalJoin']
y= data['Sentiment']
cv = TfidfVectorizer(min_df=1,stop_words='english')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

len(X_train), len(X_test), len(y_train), len(y_test)

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Logistic Regression

logreg = LogisticRegression()

logreg = logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

logreg.score(X_train,y_train)

logreg.score(X_test,y_test)

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

# Random Forest

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train);

y_pred = clf.predict(X_test)

clf.score(X_train,y_train)

clf.score(X_test,y_test)

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

# Support Vector Machine

class_linear = svm.SVC(kernel='linear')
class_linear.fit(X_train,y_train);

y_pred = class_linear.predict(X_test)

class_linear.score(X_train,y_train)

class_linear.score(X_test,y_test)

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

# Navie Bayes

mnb = MultinomialNB()

mnb.fit(X_train,y_train);

y_pred = mnb.predict(X_test);

mnb.score(X_train,y_train)

mnb.score(X_test,y_test)

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

"""**Accuracy of All 4**

1. Logistic Regression: 95.50%
2. Random Forest: 96.35%
3. Support Vector Machine: 95.75%
4. Naive Bayes: 88.25%
"""

# Saving All 4 Models to System and Vectorizer as well

pickle.dump(logreg,open("LogisticRegressionModel.pkl","wb"));

pickle.dump(clf,open("RandomForestModel.pkl","wb"));

pickle.dump(class_linear,open("SupportVectorMachineModel.pkl","wb"));

pickle.dump(mnb,open("NavieBayesModel.pkl","wb"));

pickle.dump(cv,open('vectorizer.pkl','wb'))

