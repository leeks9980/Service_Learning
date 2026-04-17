import os
import cv2 #이미지 처리를 위한 라이브러리
from matplotlib import pyplot as plt #이미지 출력을 위한 라이브러리
import numpy as np
import json  #json파일을 읽기 위한 라이브러리

img_path = r"C:\Users\leeks\OneDrive\바탕 화면\sex\labeled_apples\test\abnormal"

os.chdir(img_path)   #작업환경 변경
files_img = os.listdir(img_path) #폴더 안에 있는 것들 확인
image_name_list = []



for file in files_img:
        image_name_list.append(file)
b = 0
for img in image_name_list:
    image = cv2.imread(img)
    img_r = cv2.resize(image, (64,64))
    height, width, a = img_r.shape 
    
    zero_matrix = np.zeros((height, width))
    
    for x in range(height):
        for y in range(width):
            neighbors = [   #이웃 픽셀 리스트
                (x-1, y-1), (x-1, y), (x-1, y+1),  # 상좌, 상, 상우
                (x, y-1), (x, y+1),                # 좌, 우
                (x+1, y-1), (x+1, y), (x+1, y+1)   # 하좌, 하, 하우
            ]
            for n_x, n_y in neighbors:  #이웃 픽셀 불러오기
                if 0 <= n_x < 64 and 0 <= n_y < 64:  # 경계 체크 
                    current_pixel = img_r[x, y].astype(int) #overflow 방지용
                    neighbor_pixel = img_r[n_x, n_y].astype(int) #overflow 방지용

                    # current_pixel과 neighbor_pixel이 모두 검정색이 아닌지 확인
                    if not np.all(current_pixel == [0, 0, 0]) and not np.all(neighbor_pixel == [0, 0, 0]):
                        if sum(current_pixel)/3 <= sum(neighbor_pixel)/3 - 55:  # 이상치 탐지
                            img_r[x, y] = [0, 0, 255]
                            zero_matrix[x, y] = sum(current_pixel)

    zero_matrix = zero_matrix.reshape(-1)
    
    if 15 <= np.std(zero_matrix) < 33 and 335 <= np.var(zero_matrix) <= 1000:
        b += 1
        print('나뭇가지',img)
print(b)


