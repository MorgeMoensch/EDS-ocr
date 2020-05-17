from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X, y = digits.data / 255., digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

score_training = []
score_test = []

for nodes in range(1, 100):
    mlp = MLPClassifier(hidden_layer_sizes=(20, nodes), max_iter=100, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    score_training.insert(nodes, mlp.score(X_train, y_train))
    score_test.insert(nodes, mlp.score(X_test, y_test))

plt.title('Akkuratheit mit zunehmender Anzahl Nodes / Hidden Layer')
plt.xlabel('Anzahl Nodes pro Layer')
plt.ylabel('Akkuratheit')
plt.plot(score_training, label='Trainingsset')
plt.plot(score_test, label='Testset')
plt.legend()
plt.savefig('akkuratheit-nodes.jpg', dpi=300)


