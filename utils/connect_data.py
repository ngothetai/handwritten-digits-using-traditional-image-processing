import json
import numpy as np
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)  # Reshape thành ma trận 3D (số lượng ảnh, chiều cao, chiều rộng)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def save_features_to_json(features, output_file):
    # Chuyển đổi các đặc trưng thành một danh sách
    features_list = [feature.tolist() for feature in features]

    # Lưu danh sách đặc trưng vào file JSON
    with open(output_file, 'w') as f:
        json.dump(features_list, f)

    print('Đã lưu các đặc trưng vào tệp thành công!')

def read_features_from_json(input_file):
    # Đọc danh sách đặc trưng từ file JSON
    with open(input_file, 'r') as f:
        features_list = json.load(f)

    # Chuyển đổi danh sách đặc trưng thành mảng numpy array
    features_array = np.array(features_list)

    return features_array