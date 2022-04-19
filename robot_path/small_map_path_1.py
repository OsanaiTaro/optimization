import cv2
import numpy as np

#白紙用意
img = np.full((1050, 1700, 3), 255, dtype=np.uint8)

#path1のポイント
num = 484
cv2.rectangle(img, (num, 495), (num, 495), (0, 0, 0), thickness=-1)
for i in range(9):
    num += 80
    cv2.rectangle(img, (num, 495), (num, 495), (0, 0, 0), thickness=-1)

#画像生成
cv2.imwrite('10_robot_path_1.png', img)
