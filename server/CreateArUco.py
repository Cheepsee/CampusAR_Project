import cv2
aruco = cv2.aruco
dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = aruco.generateImageMarker(dict, 12, 600)  # 600px 方形，ID=12
cv2.imwrite("marker_12.png", img)