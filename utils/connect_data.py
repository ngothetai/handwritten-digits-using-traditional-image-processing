import json
import numpy as np
import gzip

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)  # Reshape thành ma trận 3D (số lượng ảnh, chiều cao, chiều rộng)
    print('Đã tải xong image data!')
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    print('Đã tải xong label data!')
    return data

def save_features_to_json(features, labels, output_file):
    # Tạo một danh sách chứa các đối tượng feature và label
    data = []
    for i in range(len(features)):
        feature = features[i]
        label = labels[i]
        data.append({
            'feature': feature.tolist(),
            'label': label.tolist()
        })

    # Ghi danh sách vào tệp JSON
    with open(output_file, 'w') as file:
        json.dump(data, file)

    print('Đã lưu các đặc trưng vào tệp thành công!')

def read_features_from_json(input_file):
    # Đọc danh sách đặc trưng từ file JSON
    with open(input_file, 'r') as f:
        features_list = json.load(f)

    # Chuyển đổi danh sách đặc trưng thành mảng numpy array
    features_array = np.array(features_list)

    return features_array