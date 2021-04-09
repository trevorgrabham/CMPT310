import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# read in the data

# empty lists to hold our data
title = []
body = []
combined = []
y = []
# read in all the real articles (have to do this seperately because they are in different folders)
for i in range(1500):
    with open('fake-real-news-1500/fake-real-news/train/real/real'+str(i)+'.txt',encoding="utf-8") as f:
        # the first line is always the title
        title.append(f.readline().strip())
        # this just reads the remainder of the file
        body.append(f.read())
        # combine it all back into one so that we can use all of the data later on
        combined.append(title[i] + '\n' + body[i])
        # the correct value for our ML algorithms
        y.append(1)
# read in the fake news
for i in range(1500):
    with open('fake-real-news-1500/fake-real-news/train/fake/fake'+str(i)+'.txt',encoding="utf-8") as f:
        title.append(f.readline().strip())
        body.append(f.read())
        combined.append(title[i] + '\n' +  body[i])
        y.append(0)


# parse the data

# to hold all of our parsed sentences
dictionary_title = []
stemmer = PorterStemmer()
for i in range(len(title)):
    # get rid of any characters that aren't alphabetic
    words = re.sub('[^a-zA-z]', ' ', title[i])
    # make all of our words lower case so that we don't have to worry about case sensitivity later
    words = words.lower()
    # split up all of the words around the spaces
    words = words.split()
    # apply the stemmer to the words after we filter out the words in the stop words dictionary
    words = [stemmer.stem(word) for word in words if not word in stopwords.words('english')]
    # join all of our words back together again
    words = ' '.join(words)
    # add the list of words to our dictionary
    dictionary_title.append(words)


# create the count vectorizer

countVect = CountVectorizer(max_features=5000,ngram_range=(1,2))
x_title = countVect.fit_transform(dictionary_title).toarray()


# set up and test the neural net with our final settings

net = MLPClassifier(max_iter=1000,solver='lbfgs', hidden_layer_sizes=(300,200,100))
res = cross_val_score(net,x_title,y,cv=10)
print('settings: max_iter=1000, solver=lbfgs, hidden_layer_sizes=(300,200,100)')
for i in res:
    print('accuracy: ' + str(i))
