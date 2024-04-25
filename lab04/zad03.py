import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285774)
train_inputs = train_set[:, 0:8]
train_classes = train_set[:, 8]
test_inputs = test_set[:, 0:8]
test_classes = test_set[:, 8]

scaler = StandardScaler()

train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

print("MLP (6, 3):")
mlp1 = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=500)
mlp1.fit(train_inputs, train_classes)
predictions_train = mlp1.predict(train_inputs)
print("Training accuracy:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp1.predict(test_inputs)
print("Testing accuracy:", accuracy_score(predictions_test, test_classes))
predictions = mlp1.predict(test_inputs)
conf_matrix = confusion_matrix(test_classes, predictions)
print(conf_matrix)

print("")
print("MLP (6, 3) logistic:")
mlp1 = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=500, activation="logistic")
mlp1.fit(train_inputs, train_classes)
predictions_train = mlp1.predict(train_inputs)
print("Training accuracy:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp1.predict(test_inputs)
print("Testing accuracy:", accuracy_score(predictions_test, test_classes))
predictions = mlp1.predict(test_inputs)
conf_matrix = confusion_matrix(test_classes, predictions)
print(conf_matrix)

print("")
print("MLP (6, 3, 3):")
mlp1 = MLPClassifier(hidden_layer_sizes=(6, 3, 3), max_iter=500)
mlp1.fit(train_inputs, train_classes)
predictions_train = mlp1.predict(train_inputs)
print("Training accuracy:", accuracy_score(predictions_train, train_classes))
predictions_test = mlp1.predict(test_inputs)
print("Testing accuracy:", accuracy_score(predictions_test, test_classes))
predictions = mlp1.predict(test_inputs)
conf_matrix = confusion_matrix(test_classes, predictions)
print(conf_matrix)

print("")
print("DecisionTree:")
clf = tree.DecisionTreeClassifier()
clf.fit(train_inputs, train_classes)
predictions_train = clf.predict(train_inputs)
print("Training accuracy:", accuracy_score(predictions_train, train_classes))
predictions_test = clf.predict(test_inputs)
print("Testing accuracy:", accuracy_score(predictions_test, test_classes))
predictions = clf.predict(test_inputs)
conf_matrix = confusion_matrix(test_classes, predictions)
print(conf_matrix)

print("")
print("GaussianNB:")
gnb = GaussianNB()
gnb.fit(train_inputs, train_classes)
predictions_train = gnb.predict(train_inputs)
print("Training accuracy:", accuracy_score(predictions_train, train_classes))
predictions_test = gnb.predict(test_inputs)
print("Testing accuracy:", accuracy_score(predictions_test, test_classes))
predictions = gnb.predict(test_inputs)
conf_matrix = confusion_matrix(test_classes, predictions)
print(conf_matrix)

# najlepiej sobie radzi GaussinaNB
# wiecej jest FALSE NEGATIVE
# w tym przypadku gorsze sa FN poniewaz nie wykrylismy cukrzycy u osoby chorej.