import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
import pickle

#here we loading dataset and putting it inside pandas dataframe
df  = pd.read_csv('data.csv')
#now we fill empty values with 0 
df = df.fillna(0)



#declaring our data features in x  and target in y : read codebook.txt
x=df[
    [
    'TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10'
    ,'gender','age','hand','married','familysize']
    ]


transformer = Normalizer().fit(x)
x= transformer.transform(x)
x = preprocessing.scale(x)


# y = df['nerdy']
y = df[['nerdy']]

y_transformer = Normalizer().fit(y)
y= y_transformer.transform(y)



X_train, X_test, y_train, y_test = train_test_split(
        x, y, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)
        

# logistic regression was the best algorithm with accuracy 98 : i know its an overfitting here but this is good for now
"""
skipped these steps because i saved this model in the below pickle file
"""
# clf = LogisticRegression()
# clf.fit(X_train,y_train)
# print(clf.score(X_test,y_test))
# predictions = clf.predict_proba(x)
# print(predictions)

"""
loading your model
"""
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

print(clf.predict_proba(x[0:20])*100)
"""
to save your predictions
"""
# prediction1 = pd.DataFrame(list(clf.predict_proba(x)), columns=['predictions','data']).to_csv('prediction1.csv')

"""
to save your model
""" 
# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf, f)


"""
validation and model selection
"""
# linear_scores = cross_val_score(LogisticRegression(), x, y, cv=5)
# print("LogisticRegression-Accuracy: %0.2f (+/- %0.2f)" % (linear_scores.mean(), linear_scores.std() * 2))


# linear_scores = cross_val_score(GaussianNB(), x, y, cv=5)
# print("GaussianNB-Accuracy: %0.2f (+/- %0.2f)" % (linear_scores.mean(), linear_scores.std() * 2))



