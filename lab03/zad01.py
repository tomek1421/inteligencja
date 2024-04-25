import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=285774)
# print(test_set)
# print(test_set.shape[0])
print(train_set[train_set[:, 4].argsort()])


def classify_iris(sl, sw, pl, pw):
    if sl < 7 and pw < 1:
        return("Setosa")
    elif pl > 5 and pw > 1:
        return("Virginica")
    else:
        return("Versicolor")

good_predictions = 0
len = test_set.shape[0]
for i in range(len):
    if classify_iris(*test_set[i, 0:4]) == test_set[i, 4]:
        good_predictions +=  1
print(good_predictions)
print(good_predictions/len*100, "%")