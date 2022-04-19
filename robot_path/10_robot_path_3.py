import cv2
import numpy as np

#白紙用意
img = np.full((1050, 1700, 3), 255, dtype=np.uint8)

#path3のポイント
num = 484
cv2.rectangle(img, (num, 558), (num, 558), (0, 0, 0), thickness=-1)
for i in range(9):
    num += 70
    cv2.rectangle(img, (num, 558), (num, 558), (0, 0, 0), thickness=-1)

#画像生成
cv2.imwrite('10_robot_path_3.png', img)
