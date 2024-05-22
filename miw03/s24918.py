import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# 1. Tworzenie zestawu danych
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)  # Generowanie danych z funkcji "make_moons"

# 2. Podział zestawu danych
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)  # Podział danych na zestawy treningowe i testowe

# Funkcja do rysowania krzywych ROC(czyli krzywa charakteryzujaca odbiornik, sluzy do oceny jakosci modeli klasyfikacyjnych)
def draw_roc_curve(classifier, test_data, test_labels, label, ax):
    # Sprawdzenie, czy klasyfikator moze przewidywac prawdopodobienstwa, jesli nie, uzyj funkcji decyzyjnej
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(test_data)[:, 1]
    else:
        scores = classifier.decision_function(test_data)
        probabilities = (scores - scores.min()) / (scores.max() - scores.min())  # Normalizacja wynikow funkcji decyzyjnej
    false_positive_rate, true_positive_rate, _ = roc_curve(test_labels, probabilities)  # Obliczanie FPR i TPR dla roznych progow decyzyjnych
    roc_auc = auc(false_positive_rate, true_positive_rate)  # Obliczenie pola pod krzywą ROC
    ax.plot(false_positive_rate, true_positive_rate, label=f"{label} (AUC = {roc_auc:.2f})")  # Dodanie krzywej ROC do wykresu
    ax.legend()

# 3. Klasyfikator Drzewa Decyzyjnego: Testowanie roznych kryteriow i glebokosci
decision_tree_fig, decision_tree_ax = plt.subplots()
criteria = ['gini', 'entropy']
depths = [5, 10, None]
for criterion in criteria:
    for depth in depths:
        tree_classifier = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        tree_classifier.fit(train_data, train_labels)  # Trenowanie drzewa decyzyjnego
        draw_roc_curve(tree_classifier, test_data, test_labels, f'DT {criterion.capitalize()}, Depth: {depth}', decision_tree_ax)  # Rysowanie krzywej ROC
decision_tree_ax.set_title('Krzywe ROC - Drzewa Decyzyjne')
decision_tree_ax.set_xlabel('Wskaźnik fałszywie pozytywnych (FPR)')
decision_tree_ax.set_ylabel('Wskaźnik prawdziwie pozytywnych (TPR)')

# 4. Las Losowy: Testowanie roznej liczby drzew
random_forest_fig, random_forest_ax = plt.subplots()
num_trees = [10, 50, 100]
for trees in num_trees :
    forest_classifier = RandomForestClassifier(n_estimators=trees, random_state=42)
    forest_classifier.fit(train_data, train_labels)  # Trenowanie lasu losowego
    draw_roc_curve(forest_classifier, test_data, test_labels, f'RF {trees} Trees', random_forest_ax )  # Rysowanie krzywej ROC
random_forest_ax.set_title('Krzywe ROC - Lasy Losowe')
random_forest_ax.set_xlabel('Wskaźnik fałszywie pozytywnych (FPR)')
random_forest_ax.set_ylabel('Wskaźnik prawdziwie pozytywnych (TPR)')

# 5. Regresja Logistyczna i SVM
log_reg_svm_fig, log_reg_svm_ax = plt.subplots()
logistic_regression = LogisticRegression(random_state=42)
svm_classifier = SVC(probability=True, random_state=42)
logistic_regression.fit(train_data, train_labels)  # Trenowanie regresji logistycznej
svm_classifier.fit(train_data, train_labels)  # Trenowanie SVM
draw_roc_curve(logistic_regression, test_data, test_labels, 'Regresja Logistyczna', log_reg_svm_ax)  # Rysowanie krzywej ROC dla regresji logistycznej
draw_roc_curve(svm_classifier, test_data, test_labels, 'SVM', log_reg_svm_ax)  # Rysowanie krzywej ROC dla SVM
log_reg_svm_ax.set_title('Krzywe ROC - Regresja Logistyczna i SVM')
log_reg_svm_ax.set_xlabel('Wskaźnik fałszywie pozytywnych (FPR)')
log_reg_svm_ax.set_ylabel('Wskaźnik prawdziwie pozytywnych (TPR)')

# 6. Klasyfikator Glosujacy laczacy wszystkie klasyfikatory
voting_classifier_fig, voting_classifier_ax = plt.subplots()
voting_classifier = VotingClassifier(
    estimators=[
        ('lr', logistic_regression),
        ('svm', svm_classifier),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    voting='soft'
)
voting_classifier.fit(train_data, train_labels)  # Trenowanie klasyfikatora glosujacego
draw_roc_curve(voting_classifier, test_data, test_labels, 'Klasyfikator Głosujący', voting_classifier_ax)  # Rysowanie krzywej ROC
voting_classifier_ax.set_title('Krzywa ROC - Klasyfikator Głosujący')
voting_classifier_ax.set_xlabel('Wskaźnik fałszywie pozytywnych (FPR)')
voting_classifier_ax.set_ylabel('Wskaźnik prawdziwie pozytywnych (TPR)')

plt.show()