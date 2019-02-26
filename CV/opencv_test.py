import cv2

print(cv2.__version__)
image = cv2.imread("E:/data/Pet_Color/Pet_Color(64x64)/cat01_64.png", cv2.IMREAD_UNCHANGED)
cv2.imshow("Moon", image)
cv2.waitKey(0)
