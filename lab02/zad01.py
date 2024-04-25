import pandas as pd

missing_values = ["NA", "-", "nan"]
df = pd.read_csv("iris_with_errors.csv", na_values = missing_values)
# print(df.values)

# a)
nullsAmount = df.isnull().sum()
print(f"Nieuzupełnione dane:\n{nullsAmount}")

# b)

for column in df.loc[:, df.columns != 'variety']: # [wiersze, kolumny]
    df.loc[(df[column] > 15) | (df[column] < 0) , column] = df[column].median()

print(df.values)

# c)
# print(df.loc[~df['variety'].isin(["Setosa", "Virginica", "Versicolor"]), 'variety'])

df.loc[df['variety'] == "setosa", 'variety'] = "Setosa"
df.loc[df['variety'] == "virginica", 'variety'] = "Virginica"
df.loc[df['variety'] == "Versicolour", 'variety'] = "Versicolor"

# print(df.loc[~df['variety'].isin(["Setosa", "Virginica", "Versicolor"]), 'variety'])