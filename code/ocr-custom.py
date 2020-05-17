from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
import numpy as np

digits = datasets.load_digits()

X, y = digits.data / 255., digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(20, 5), max_iter=100, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
mlp.fit(X_train, y_train)

for i in range(1, 10):
    img = io.imread('../digits/{}.png'.format(i), as_gray=True)
    img = resize(img, (8,8))
    img = np.reshape(img, [-1])
    print()
    print("Predicted Number: ", mlp.predict([img])[0], "Actual Number: ", i)
