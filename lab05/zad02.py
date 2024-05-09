import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import History
from sklearn.metrics import confusion_matrix

# a) Preprocessing:
#    - reshape: zmienia kształt danych, np. z tablicy 1D na 2D (np. z obrazów o rozmiarze 28x28 na 28x28x1).
#    - to_categorical: przekształca etykiety klas na postać one-hot encoding.
#    - np.argmax: odwrócenie procesu one-hot encoding, zamieniając zakodowane etykiety z powrotem na wartości liczbowe.

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)

# b) Dane przepływają przez sieć neuronową:
#    - Wejściowa warstwa konwolucyjna (Conv2D) otrzymuje obrazy 28x28x1.
#    - MaxPooling2D zmniejsza wymiary obrazu.
#    - Warstwa Flatten konwertuje dane do postaci jednowymiarowej.
#    - Warstwa Dense przyjmuje te dane i wykonuje operacje neuronowe.
#    - Na wyjściu otrzymujemy prawdopodobieństwa przynależności do poszczególnych klas.

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# c) Liczymy macierz błędów (confusion matrix) i analizujemy, które cyfry są mylone z innymi.
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# d) Krzywe uczenia się pomagają ocenić, czy model jest dobrze dopasowany, niedouczony czy przeuczony.
#    Jeśli krzywe trenowania i walidacji zbliżają się do siebie i stabilizują się na wysokim poziomie, to model jest dobrze dopasowany.
#    Jeśli krzywa treningowa rośnie, a krzywa walidacji spada, to model jest przeuczony.
#    Jeśli obie krzywe pozostają niskie, model jest niedouczony.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# e) Aby zapisać model do pliku .h5 co epokę po osiągnięciu lepszego wyniku, możemy dodać warunek zapisu do pętli trenującej.
#    Na przykład, możemy przechowywać dotychczasową najlepszą wartość metryki i porównywać ją z wynikiem każdej epoki.
#    Jeśli wynik jest lepszy, zapisujemy model.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()


# Często 5 jest odczytywana jako 8,  8 jako 9 i 3 jako 5.
# Nie mamy do do czynienia ani z przeuczeniem ani z niedouczeniem

