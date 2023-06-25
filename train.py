from utils.feature_extract import *
from utils.connect_data import *
from tqdm import tqdm

X_train = load_mnist_images('data/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')

features = []
for image in tqdm(X_train):
    features.append(extract_hog_features(image))
print('Trích xuất đặc trưng HOG thành công!!!')
save_features_to_json(features, y_train, 'extract_data/train_HOG_features.json')
print('Lưu đặc trưng thành công!!!')