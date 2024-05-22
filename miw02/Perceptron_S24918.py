import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=50):
        """
        Inicjalizuje perceptron z określonym współczynnikiem uczenia i liczbą iteracji

        Parametry:
        - learning_rate: Współczynnik uczenia
        - n_iterations: Liczba iteracji w procesie uczenia
        """
        self.weights = None
        self.learning_rate = learning_rate  # Ustawia współczynnik uczenia
        self.n_iterations = n_iterations  # Ustawia liczbę iteracji uczenia

    def train(self, X, y):
        """
        Trenuje perceptron na podstawie podanych danych

        Parametry:
        - X: Macierz cech, gdzie każdy wiersz to jedna obserwacja
        - y: Wektor etykiet odpowiadających obserwacjom w X
        """
        self.weights = np.zeros(X.shape[1] + 1)  # Inicjalizuje wagi z zerami
        for _ in range(self.n_iterations):  # Pętla iteracji uczenia
            for xi, target in zip(X, y):  # Iteracja po każdej obserwacji
                # Aktualizacja wag na podstawie błędu przewidywania
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi  # Aktualizacja wag cech
                self.weights[0] += update  # Aktualizacja wagi biasu (interceptu)

    def predict(self, X):
        """
        Przewiduje klasy na podstawie nauczonych wag

        Parametry:
        - X: Macierz cech, której klasy są przewidywane

        Zwraca:
        - Wektor przewidywanych klas
        """
        # Obliczenie aktywacji jako kombinacja liniowa cech i wag
        activation = np.dot(X, self.weights[1:]) + self.weights[0]
        # Przewidywanie klasy na podstawie aktywacji
        return np.where(activation >= 0, 1, -1)

class MultiClassPerceptron:
    def __init__(self, mode='OvR', learning_rate=0.1, n_iterations=50):
        """
        Inicjalizuje perceptron wieloklasowy z określonym trybem, współczynnikiem uczenia i liczbą iteracji

        Parametry:
        - mode: Tryb klasyfikacji ('OvR' dla jeden-vs-reszta lub 'OvO' dla jeden-na-jednego)
        - learning_rate: Współczynnik uczenia
        - n_iterations: Liczba iteracji w procesie uczenia
        """
        self.unique_classes = None
        self.learning_rate = learning_rate  # Współczynnik uczenia
        self.n_iterations = n_iterations  # Liczba iteracji uczenia
        self.mode = mode  # Tryb klasyfikacji
        self.classifiers = []  # Lista klasyfikatorów perceptronu

    def train(self, X, y):
        """
        Trenuje klasyfikatory perceptronu w trybie wieloklasowym

        Parametry:
        - X: Macierz cech treningowych
        - y: Wektor etykiet klas dla cech treningowych
        """
        self.unique_classes = np.unique(y)  # Znajduje unikalne klasy w etykietach
        if self.mode == 'OvR':
            # Trenowanie jednego perceptronu na klasę (jeden przeciwko reszcie)
            for cls in self.unique_classes:
                y_binary = np.where(y == cls, 1, -1)  # Binarna transformacja etykiet
                classifier = Perceptron(self.learning_rate, self.n_iterations)
                classifier.train(X, y_binary)  # Trenowanie perceptronu
                self.classifiers.append(classifier)  # Dodawanie klasyfikatora do listy
        else:  # Tryb OvO (jeden na jednego)
            for i, class1 in enumerate(self.unique_classes):
                for class2 in self.unique_classes[i + 1:]:
                    # Filtrowanie danych tylko dla dwóch klas
                    X_filtered = X[(y == class1) | (y == class2)]
                    y_filtered = y[(y == class1) | (y == class2)]
                    y_binary = np.where(y_filtered == class1, 1, -1)
                    classifier = Perceptron(self.learning_rate, self.n_iterations)
                    classifier.train(X_filtered, y_binary)  # Trenowanie perceptronu dla pary klas
                    self.classifiers.append((classifier, class1, class2))  # Dodawanie klasyfikatora do listy

    def predict(self, X):
        """
        Przewiduje klasy dla danych wejściowych w trybie wieloklasowym

        Parametry:
        - X: Macierz cech, dla której są przewidywane klasy

        Zwraca:
        - Wektor przewidywanych klas dla danych wejściowych
        """
        if self.mode == 'OvR':
            # Zbiera przewidywania od wszystkich klasyfikatorów OvR
            predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
            return np.argmax(predictions, axis=0)  # Wybiera klasę z najwyższym wynikiem
        else:  # Tryb OvO
            votes = np.zeros((X.shape[0], len(self.unique_classes)))  # Macierz głosów
            for classifier, class1, class2 in self.classifiers:
                predictions = classifier.predict(X)
                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        votes[i, class1] += 1  # Głos na class1
                    else:
                        votes[i, class2] += 1  # Głos na class2
            return np.argmax(votes, axis=1)  # Wybiera klasę z największą liczbą głosów

def generate_data(n_clusters=4, points_per_cluster=50, n_features=2, random_state=None):
    """
    Generuje losowe dane z określoną liczbą klastrów, punktów na klaster i cech

    Parametry:
    - n_clusters: Liczba klastrów
    - points_per_cluster: Liczba punktów na klaster
    - n_features: Liczba cech każdego punktu
    - random_state: Ziarno dla generatora liczb losowych

    Zwraca:
    - X: Wygenerowana macierz cech
    - y: Wektor etykiet klas dla wygenerowanych danych
    """
    np.random.seed(random_state)  # Ustawia ziarno dla powtarzalności
    data, labels = [], []
    for _ in range(n_clusters):
        center = np.random.uniform(-10, 10, n_features)  # Losowe centrum dla klastra
        data.append(center + np.random.randn(points_per_cluster, n_features))  # Punkty wokół centrum
        labels.extend([_] * points_per_cluster)  # Etykiety dla punktów
    return np.vstack(data), np.array(labels)

def split_data(X, y, train_size=0.8, random_state=None):
    """
    Dzieli dane na zbiory treningowe i testowe

    Parametry:
    - X: Macierz cech
    - y: Wektor etykiet
    - train_size: Proporcja danych treningowych
    - random_state: Ziarno dla generatora liczb losowych

    Zwraca:
    - X_train, X_test: Podzielone macierze cech
    - y_train, y_test: Podzielone wektory etykiet
    """
    np.random.seed(random_state)  # Ustawia ziarno dla powtarzalności
    indices = np.random.permutation(len(X))  # Losowa permutacja indeksów
    split_point = int(len(X) * train_size)  # Punkt podziału
    # Dzieli dane i etykiety na podstawie punktu podziału
    return X[indices[:split_point]], X[indices[split_point:]], y[indices[:split_point]], y[indices[split_point:]]

# Przykład użycia
X, y = generate_data()  # Generuje dane
X_train, X_test, y_train, y_test = split_data(X, y)  # Dzieli dane

# Inicjalizacja klasyfikatorów
ovr_classifier = MultiClassPerceptron('OvR')
ovo_classifier = MultiClassPerceptron('OvO')

# Trenowanie i ocena klasyfikatorów
for classifier in [ovr_classifier, ovo_classifier]:
    classifier.train(X_train, y_train)  # Trenowanie klasyfikatora
    # Obliczanie dokładności na zbiorze treningowym i testowym
    train_acc = np.mean(classifier.predict(X_train) == y_train)
    test_acc = np.mean(classifier.predict(X_test) == y_test)
    print(f"Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Wizualizacja granic decyzyjnych
def plot_decision_boundary(classifier, X, y, title='Decision Boundary'):
    """
    Rysuje granicę decyzyjną klasyfikatora wraz z danymi

    Parametry:
    - classifier: Klasyfikator do wizualizacji
    - X: Macierz cech
    - y: Wektor etykiet
    - title: Tytuł wykresu
    """
    # Ustala zakres dla cech
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Tworzy siatkę punktów
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # Przewiduje klasy dla każdego punktu siatki
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Rysuje granicę decyzyjną i dane
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# Wizualizacja dla klasyfikatorów OvR i OvO
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plot_decision_boundary(ovr_classifier, X_train, y_train, 'OvR Decision Boundary')
plt.subplot(1, 2, 2)
plot_decision_boundary(ovo_classifier, X_train, y_train, 'OvO Decision Boundary')
plt.tight_layout()
plt.show()
