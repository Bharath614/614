
# 1. K-Nearest Neighbours (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# KNN Implementation
def knn_example():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred))

# Naive Bayes
def naive_bayes_example():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))

# Logistic Regression
def logistic_regression_example():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    print("Logistic Regression Score:", model.score(X_test, y_test))

# Linear Regression
def linear_regression_example():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.scatter(X, y)
    plt.plot(X, y_pred, color='red')
    plt.title("Linear Regression")
    plt.show()

# Linear vs Polynomial Regression
def linear_vs_polynomial():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([1, 4, 9, 16, 25])
    linear = LinearRegression()
    linear.fit(X, y)
    y_pred_linear = linear.predict(X)
    poly_feat = PolynomialFeatures(degree=2)
    X_poly = poly_feat.fit_transform(X)
    poly = LinearRegression()
    poly.fit(X_poly, y)
    y_pred_poly = poly.predict(X_poly)
    plt.scatter(X, y)
    plt.plot(X, y_pred_linear, label="Linear", color='blue')
    plt.plot(X, y_pred_poly, label="Polynomial", color='red')
    plt.legend()
    plt.title("Linear vs Polynomial Regression")
    plt.show()

# Expectation Maximization using GMM
def em_example():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.6)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X)
    labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("Expectation Maximization using GMM")
    plt.show()

# Credit Score Classification
def credit_score_classification():
    data = pd.DataFrame({
        'income': [50000, 60000, 100000, 30000, 25000],
        'loan_amount': [10000, 20000, 30000, 15000, 10000],
        'credit_score': [700, 800, 650, 500, 400],
        'label': [1, 1, 1, 0, 0]
    })
    X = data[['income', 'loan_amount', 'credit_score']]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

# Car Price Prediction
def car_price_prediction():
    data = pd.DataFrame({
        'mileage': [10, 20, 30, 40],
        'price': [200000, 180000, 160000, 140000]
    })
    X = data[['mileage']]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Predicted Price for mileage 25:", model.predict([[25]]))

# House Price Prediction
def house_price_prediction():
    df = pd.DataFrame({
        'area': [1000, 1500, 1800, 2000, 2500],
        'bedrooms': [2, 3, 3, 4, 4],
        'price': [300000, 400000, 450000, 500000, 600000]
    })
    X = df[['area', 'bedrooms']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Predicted house price:", model.predict([[1600, 3]]))

# Compare Classification Algorithms
def compare_classifiers():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    models = {
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=200)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Mobile Price Prediction
def mobile_price_prediction():
    df = pd.DataFrame({
        'RAM': [2, 4, 6, 8],
        'Storage': [32, 64, 128, 256],
        'Price': [8000, 12000, 18000, 25000]
    })
    X = df[['RAM', 'Storage']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Predicted Mobile Price:", model.predict([[6, 128]]))

# Perceptron for IRIS
def perceptron_example():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    model = Perceptron()
    model.fit(X_train, y_train)
    print("Perceptron Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Bank Loan Prediction
def bank_loan_prediction():
    df = pd.DataFrame({
        'income': [50000, 60000, 30000, 25000],
        'loan_amount': [10000, 20000, 15000, 10000],
        'approved': [1, 1, 0, 0]
    })
    X = df[['income', 'loan_amount']]
    y = df['approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Loan Approval Prediction:", model.predict([[45000, 12000]]))

# Future Sales Prediction
def future_sales_prediction():
    months = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    sales = np.array([100, 120, 130, 150, 170])
    model = LinearRegression()
    model.fit(months, sales)
    print("Predicted Sales for Month 6:", model.predict(np.array([[6]])))

#ID3 classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Manual encoding:
# Outlook: Sunny=0, Overcast=1, Rain=2
# Temp: Hot=0, Mild=1, Cool=2
# Humidity: High=0, Normal=1
# Windy: False=0, True=1
# Target: No=0, Yes=1

X = [
    [0, 0, 0, 0],   
    [0, 0, 0, 1],   
    [1, 0, 0, 0],  
    [2, 1, 0, 0],  
]
y = [0, 0, 1, 1]
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Plot tree
plot_tree(model)
plt.show()

#Artificial Neural Network
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Use XOR logic to get non-linear learning
x = [[0, 0], [0, 2], [0, 0], [2, 1]]
y = [2, 1, 1, 0]

model = MLPClassifier(hidden_layer_sizes=(2,))
model.fit(x, y)
pred = model.predict(x)
print("Predictions:", pred)

# Plot predictions (this still gives flat line unless you train over epochs)
plt.plot(x, pred, marker='o')
plt.title("Predicted Output for XOR")
plt.xlabel("Sample")
plt.ylabel("Prediction")
plt.grid(True)
plt.show()