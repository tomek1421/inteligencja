from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# loading data
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# print(iris_df.describe())

# # original
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], c=iris_df['species'])
plt.title('Original Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Min-Max normalised
scaler = MinMaxScaler()
iris_minmax = iris_df.copy()
iris_minmax[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(iris_df[['sepal length (cm)', 'sepal width (cm)']])
plt.scatter(iris_minmax['sepal length (cm)'], iris_minmax['sepal width (cm)'], c=iris_minmax['species'])
plt.title('Min-Max Normalised Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Z-Score scaled
scaler = StandardScaler()
iris_scaled = iris_df.copy()
iris_scaled[['sepal length (cm)', 'sepal width (cm)']] = scaler.fit_transform(iris_df[['sepal length (cm)', 'sepal width (cm)']])
plt.scatter(iris_scaled['sepal length (cm)'], iris_scaled['sepal width (cm)'], c=iris_scaled['species'])
plt.title('Z-Score Scaled Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Min-Max
# Oryginalny plot zawiera faktyczne min i max naszych danych
# Plot Min-Max zawiera dane z zakresu 0-1
# Plot Z-Score, dla średniej przypada 0 a odchylenie na 1 więc dane układane są w zależności od tego jak bardzo są odchylone
# od średniej

# Mean
# Oryginalny plot zawiera średnią długość i średnią szerokość sepali
# Plot Min-Max, brak konkretnej interpretacji, pomaga skalować dane pomiędzy 0 a 1

# SD
# Oryginalny plot ukazuje odchylanie pomiedzy danymi
# Plot Min-Max, brak konkretnej interpretacji, pomaga skalować dane
# Plot Z-Score, dane koncentrują się wokół 0 i rozpraszają ułatwiając porównanie j