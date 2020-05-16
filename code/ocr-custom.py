from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
import numpy as np

digits = datasets.load_digits()

X, y = digits.data / 255., digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

custom_digits = []
custom_validation = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in range(1, 9):
    img = io.imread('../digits/{}.png'.format(i), as_gray=True)
    print(img)
    custom_digits.append(img[0])
    print(custom_digits)



mlp = MLPClassifier(hidden_layer_sizes=(20, 5), max_iter=100, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
mlp.fit(X_train, y_train)
print(mlp.score(custom_digits, custom_validation))

