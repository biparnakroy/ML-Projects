import pandas as pd 
import quandl , math , datetime
import numpy as np 
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']= (df['Adj. High']- df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change']= (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT','PCT_change', 'Adj. Volume']]


forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))

print(forcast_out)

df['label']=df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
x=X[:-forcast_out]
X_lately = X[-forcast_out:]
x = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['label'])

x_train , x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)

with open('LinearRegression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('LinearRegression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)

forcast_set = clf.predict(X_lately)

print(forcast_set, accuracy, forcast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()