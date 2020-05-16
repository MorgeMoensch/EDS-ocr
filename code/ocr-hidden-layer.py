from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X, y = digits.data / 255., digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

score_trainingset = []
score_testset = []

for hls in range(1, 100):
    mlp = MLPClassifier(hidden_layer_sizes=hls, max_iter=100, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    score_trainingset.insert(hls, mlp.score(X_train, y_train))
    score_testset.insert(hls, mlp.score(X_test, y_test))

plt.title('Akkuratheit mit zunehmender Anzahl Hidden Layers')
plt.xlabel('Anzahl Hidden Layer')
plt.ylabel('Akkuratheit')
plt.plot(score_trainingset, label='Trainingsset')
plt.plot(score_testset, label='Testset')
plt.legend()
plt.savefig('akkuratheit-hidden-layer.jpg', dpi = 300)
