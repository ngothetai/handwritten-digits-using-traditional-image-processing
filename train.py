from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
train_images = mnist.data[:60000]
train_labels = mnist.target[:60000]
test_images = mnist.data[60000:]
test_labels = mnist.target[60000:]