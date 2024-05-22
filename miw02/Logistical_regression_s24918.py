import numpy as np
import matplotlib.pyplot as plt

def calculate_softmax(z):
    # Zastosowanie funkcji softmax do znormalizowania logitów do prawdopodobieństw
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        self.intercept = None
        self.coefs = None
        self.unique_classes = None
        self.rate = learning_rate  # Współczynnik uczenia
        self.iterations = n_iterations  # Liczba iteracji algorytmu spadku gradientu

    def fit(self, X, y):
        samples, attributes = X.shape
        self.unique_classes = np.unique(y)
        classes_count = len(self.unique_classes)

        # Inicjalizacja wag i biasów dla każdej klasy. Każda kolumna odpowiada oddzielnemu klasyfikatorowi regresji logistycznej dla jednej klasy
        self.coefs = np.zeros((attributes, classes_count))  # Wagi
        self.intercept = np.zeros(classes_count)  # Biasy

        # Konwersja etykiet celów na formę zakodowaną "one-hot"
        targets_encoded = np.eye(classes_count)[y]

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.coefs) + self.intercept
            predicted_probs = calculate_softmax(linear_model)

            # Obliczenie gradientów dla wag i biasów
            gradient_w = (1 / samples) * np.dot(X.T, predicted_probs - targets_encoded)
            gradient_b = (1 / samples) * np.sum(predicted_probs - targets_encoded, axis=0)

            # Aktualizacja wag i biasów
            self.coefs -= self.rate * gradient_w
            self.intercept -= self.rate * gradient_b

    def predict(self, X):
        # Obliczenie prawdopodobieństw klas i zwrócenie klasy o najwyższym prawdopodobieństwie
        linear_output = np.dot(X, self.coefs) + self.intercept
        predicted_probs = calculate_softmax(linear_output)
        return np.argmax(predicted_probs, axis=1)

    def model_accuracy(self, X, y):
        # Obliczenie dokładności jako ułamek poprawnie przewidzianych instancji
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Poniższe funkcje służą do tworzenia syntetycznych danych, podziału danych i wizualizacji granic decyzyjnych
def create_data(points=50, groups=4, attributes=2):
    data_features = np.zeros((points * groups, attributes))
    data_labels = np.zeros(points * groups)
    for group in range(groups):
        data_features[group * points:(group + 1) * points] = np.random.randn(points, attributes) + np.random.randn(
            attributes) * 5
        data_labels[group * points:(group + 1) * points] = group
    return data_features, data_labels.astype(int)

def partition_data(features, labels, ratio=0.8):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(features.shape[0])
    partition = int(len(features) * ratio)
    training_features, testing_features = features[shuffled_indices[:partition]], features[shuffled_indices[partition:]]
    training_labels, testing_labels = labels[shuffled_indices[:partition]], labels[shuffled_indices[partition:]]
    return training_features, testing_features, training_labels, testing_labels

def visualize_boundaries(features, labels, classifier):
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
    predictions = classifier.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    predictions = predictions.reshape(grid_x.shape)
    plt.contourf(grid_x, grid_y, predictions, alpha=0.5, levels=np.arange(len(np.unique(labels)) + 1) - 0.5,
                 cmap='Paired')
    plt.colorbar(ticks=np.unique(labels))
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolor='k')
    plt.xlabel('Atrybut 1')
    plt.ylabel('Atrybut 2')
    plt.title('Wizualizacja granic decyzyjnych')

# Główne wykonanie: generacja danych, podział, trening modelu i wizualizacja
feature_set, label_set = create_data()
features_train, features_test, labels_train, labels_test = partition_data(feature_set, label_set)

classifier = SoftmaxRegression()
classifier.fit(features_train, labels_train)

accuracy_train = classifier.model_accuracy(features_train, labels_train)
accuracy_test = classifier.model_accuracy(features_test, labels_test)
print(f"Dokładność na zbiorze treningowym: {accuracy_train * 100:.2f}%")
print(f"Dokładność na zbiorze testowym: {accuracy_test * 100:.2f}%")

plt.figure(figsize=(10, 6))
visualize_boundaries(feature_set, label_set, classifier)
plt.show()
