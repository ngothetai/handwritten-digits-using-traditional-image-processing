from skimage.feature import hog
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure

def calculate_lbp_histogram(image, num_points, radius):
    # Chuyển ảnh sang ảnh xám nếu cần
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Chuẩn bị một mảng histogram có 256 bins (0-255)
    histogram = np.zeros(256, dtype=np.int32)
    
    # Lấy kích thước ảnh
    height, width = gray.shape
    
    # Duyệt qua từng pixel trong ảnh (ngoại trừ cạnh)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Lấy giá trị của tâm
            center = gray[y, x]
            
            # Khởi tạo một mảng 8 điểm láng giềng
            neighbors = np.zeros(8, dtype=np.uint8)
            
            # Lấy giá trị của 8 điểm láng giềng (theo chiều kim đồng hồ, bắt đầu từ 12 giờ)
            neighbors[0] = gray[y - 1, x]
            neighbors[1] = gray[y - 1, x + 1]
            neighbors[2] = gray[y, x + 1]
            neighbors[3] = gray[y + 1, x + 1]
            neighbors[4] = gray[y + 1, x]
            neighbors[5] = gray[y + 1, x - 1]
            neighbors[6] = gray[y, x - 1]
            neighbors[7] = gray[y - 1, x - 1]
            
            # So sánh giá trị của tâm với các điểm láng giềng
            lbp_code = 0
            for i in range(8):
                if neighbors[i] >= center:
                    lbp_code |= (1 << i)
            
            # Tính toán histogram
            histogram[lbp_code] += 1
    
    # Chuẩn hóa histogram thành vector đặc trưng (tổng các bin = 1)
    histogram = histogram / np.sum(histogram)
    #print('Trích xuất LBP xong!')
    return histogram


def extract_hog_features(image):
    # Chuyển ảnh sang ảnh xám nếu cần
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuẩn hóa ảnh đầu vào
    image = exposure.rescale_intensity(image, in_range=(0, 255))

    # Trích xuất đặc trưng HOG
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)

    # # Chuẩn hóa đặc trưng HOG
    features = features / np.linalg.norm(features)

    #print('Trích xuất HOG xong!')
    return features


def Otsu(image):
    # Chuyển ảnh sang ảnh xám nếu cần
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Áp dụng phương pháp Otsu để tìm ngưỡng tự động
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold