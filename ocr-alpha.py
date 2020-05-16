from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X, y = digits.data / 255., digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

score_trainingset = []
score_testset = []
alpha_axis = []

alpha = 1e-50
while(alpha < 100):
    mlp = MLPClassifier(hidden_layer_sizes=(20, 6), max_iter=100, alpha=alpha, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    alpha_axis.append(alpha)
    score_trainingset.append(mlp.score(X_train, y_train))
    score_testset.append(mlp.score(X_test, y_test))
    alpha = alpha * 10

plt.title('Akkuratheit mit abnehmendem Alpha-Wert')
plt.xlabel('Alpha-Wert')
plt.xscale=('log')
plt.ylabel('Akkuratheit')
plt.plot(alpha_axis, score_trainingset, label='Trainingsset')
plt.plot(alpha_axis, score_testset, label='Testset')
plt.legend()
plt.savefig('akkuratheit-alpha-3.jpg', dpi = 300)
