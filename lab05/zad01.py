import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

iris = load_iris()
X = iris.data
y = iris.target

# Standaryzuje dane usuwając średnią i odchylenie
# Wzorem jest z = (x - u) / s, gdzie u to średnia a s odchylenie
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# OneHotEncoder zamienia etykiety klas na binarny wektor, w którym każda wartość oznacza przynależność do kategorii (1 jeśli należy).
# Etykiety klas tworzą razem macierz.
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Liczba neuronów warstwy wejściowej zależy od liczby kolumn danych 
# treningowych, gdyż taką właśnie liczbę zwraca X_train.shape[1]
# y_encoded.shaper[1] zwraca liczbę klas w danych, więc taką liczbę
# neuronów będzie miała warstwa wyjściowa

# Relu i tanh ma 100% accuracy
model = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(64, activation='tanh'),
    Dense(y_encoded.shape[1], activation='softmax')
])


# Optymizatory rmsprop i adam są tak samo szybkie i oba mają 100% dokładności
# sgd jest zauważalnie szybszy, ale zmniejsza dokładność do około 80%
# adagrad nie jest szybszy a zmiejsza dokładność do około 70%
# w opptymalizatorze można zmienić szybkość uczenia
from keras.optimizers import Adam
custom_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Im mniejszy rozmiar patii, tym jest większa różnica pomiędzy
# train loss a validation loss. To samo z train accuracy a validation accuracy
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Najlepszą wydajność sieć osiągnęła w epoce 100
# Krzywa sugeruje dobrze dopasowany model, ponieważ validation loss nie wzrasta i validation accuracy nie spada
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

model.save('iris_model.h5')
