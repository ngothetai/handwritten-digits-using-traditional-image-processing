from skimage.feature import hog
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    
    return histogram


def extract_hog_features(image_path):
    # Đọc ảnh và chuyển đổi sang ảnh xám
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    
    # Trích chọn đặc trưng HOG
    features, hog_image = hog(gray_image, visualize=True)
    
    # Hiển thị ảnh gốc và ảnh HOG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Image')
    ax2.axis('off')
    plt.show()
    
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