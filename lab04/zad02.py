import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285774)
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

scaler = StandardScaler()

train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

mlp1 = MLPClassifier(hidden_layer_sizes=(2), max_iter=3000)
mlp1.fit(train_inputs, train_classes)
print("1 warstwa 2 neurony:")
predictions_train = mlp1.predict(train_inputs)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp1.predict(test_inputs)
print(accuracy_score(predictions_test, test_classes))

mlp2 = MLPClassifier(hidden_layer_sizes=(3), max_iter=3000)
mlp2.fit(train_inputs, train_classes)
print("1 warstwa 3 neurony:")
predictions_train = mlp2.predict(train_inputs)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp2.predict(test_inputs)
print(accuracy_score(predictions_test, test_classes))

mlp3 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=3000)
mlp3.fit(train_inputs, train_classes)
print("2 warstwy 3 neurony:")
predictions_train = mlp3.predict(train_inputs)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp3.predict(test_inputs)
print(accuracy_score(predictions_test, test_classes))

#zwykle najlepiej radzi sobie model 3