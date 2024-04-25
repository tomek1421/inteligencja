import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285774)
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

#Ewaluacja
scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.fit_transform(test_inputs)

for n in [3, 5, 11]:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train_inputs, train_classes)
    print("")
    print(f"{n} neighbours:")
    print("Accuracy:", clf.score(test_inputs, test_classes))
    predictions = clf.predict(test_inputs)
    print("Confusion matrix:")
    print(confusion_matrix(test_classes, predictions))

gnb = GaussianNB()
gnb.fit(train_inputs, train_classes)
print("")
print("gaussian naive bayes:")
print("Accuracy:", gnb.score(test_inputs, test_classes))
predictions = gnb.predict(test_inputs)
print("Confusion matrix:")
print(confusion_matrix(test_classes, predictions))

#Najlepiej wypadł klasyfikator 11NN