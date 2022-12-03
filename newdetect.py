import cv2
import numpy as np


# img = cv2.imread('temp.jpg')
# cv2.imshow('img', img)

# black = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray = cv2.GaussianBlur(gray, (9, 9), 20)
# cv2.imshow('gray ', gray)

# thres = 127
# n, m = gray.shape
# for i in range(n):
#     for j in range(m):
#         if (gray[i][j] > thres):
#             gray[i][j] = 255
#         else:
#             gray[i][j] = 0


# cv2.imshow('thres', gray)

# contours, _ = cv2.findContours(
#     gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contours = sorted(contours, key=cv2.contourArea, reverse=True)

# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 15, True)
#     if len(approx) == 4:
#         cv2.drawContours(
#             black, cnt, -1, (0, 255, 0), 3)

# cv2.imshow("81 ", black)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


vid = cv2.VideoCapture(0)
while (True):
    ret, img = vid.read()
    # img = cv2.imread('temp.jpg')
    cv2.imshow('img', img)

    black = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 20)

    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imshow('thres', gray)

    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 15, True)
        if len(approx) == 4:
            cv2.drawContours(
                black, cnt, -1, (0, 255, 0), 3)

    cv2.imshow("81 ", black)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
