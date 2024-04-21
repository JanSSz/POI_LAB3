import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
feature_vectors_file = "C:\\Users\\Hyperbook\\Desktop\\output_directory\\feature_vectors.csv"
df = pd.read_csv(feature_vectors_file)

# Konwersja stringów na listy liczb
df['Dissimilarity'] = df['Dissimilarity'].apply(lambda x: float(x.strip('[]').split()[0]))
df['Correlation'] = df['Correlation'].apply(lambda x: float(x.strip('[]').split()[0]))
df['Contrast'] = df['Contrast'].apply(lambda x: float(x.strip('[]').split()[0]))
df['Energy'] = df['Energy'].apply(lambda x: float(x.strip('[]').split()[0]))
df['Homogeneity'] = df['Homogeneity'].apply(lambda x: float(x.strip('[]').split()[0]))
df['ASM'] = df['ASM'].apply(lambda x: float(x.strip('[]').split()[0]))

# Podział danych na zbiór cech (X) i etykiety (Y)
X = df.drop(['Category', 'File'], axis=1)
Y = df['Category']

# Odczyt unikalnych wartości klas z kolumny 'Category'
class_labels = Y.unique()

# Utworzenie i uczenie klasyfikatora SVM
clf = SVC(gamma='auto')

# Podział danych na zbiór treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)  # dodanie stratyfikacji do balansu

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
#generowanie macierzy pomyłek
cm = confusion_matrix(y_test, y_pred, normalize='true')
print("Confusion Matrix:")
print(cm)

# Wyświetlenie macierzy pomyłek
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
