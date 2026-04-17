import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# 사전학습된 ResNet50 모델 (분류층 제외)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 이미지 전처리 함수
def load_and_preprocess(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# 이미지 폴더 경로
image_dir = r"C:\Users\leeks\OneDrive\바탕 화면\sex\labeled_apples\test\abnormal"
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

# 특징 벡터 추출
features = []
for path in image_paths:
    x = load_and_preprocess(path)
    feat = model.predict(x)
    features.append(feat.flatten())
features = np.array(features)

# PCA 차원 축소 (2차원)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# KMeans 클러스터링 (클러스터 개수 3으로 설정)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_2d)

# 결과 시각화
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters)
plt.title('Image Clusters after PCA + KMeans')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
