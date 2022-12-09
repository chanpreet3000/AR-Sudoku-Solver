import cv2
import numpy as np
import tensorflow
import solver

input_size = 48
height = 900
width = 900


def check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (height, width))
    _, thres = cv2.threshold(gray, 100, 255, 0)
    # bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    # edged = cv2.Canny(bfilter, 5, 180)
    contours, _ = cv2.findContours(
        thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("81 ", cv2.drawContours(
        img, contours, -1, (0, 255, 0), 3))
    num = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            num = num + 1

    if num >= 70:
        return True
    else:
        return False


def get_InvPerspective(img, masked_num, location):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(
        masked_num, matrix, (img.shape[1], img.shape[0]))
    return result


def display_numbers(img, numbers, color=(0, 0, 255)):
    W = int(height/9)
    H = int(width/9)
    for i in range(9):
        for j in range(9):
            if numbers[(j*9)+i] != 0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int(
                    (j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img


def split_boxes(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    return boxes


def get_perspective(img, location):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def predict_sudoku_boxes(rois):
    prediction = model.predict(rois)
    predicted_numbers = []
    for i in prediction:
        index = (np.argmax(i))
        predicted_number = index
        predicted_numbers.append(predicted_number)

    # reshape the list
    return predicted_numbers


def solveSudoku(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 5)
    cv2.imshow("GRAY ", gray)

    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    cv2.imshow("Testing ", cv2.drawContours(
        img, contours, -1, (0, 255, 0), 3))
    num = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            num = num + 1

            warp_img = get_perspective(img, approx)
            warp_img = cv2.resize(warp_img, (height, width))

            for i in range(4):
                warp_img = cv2.rotate(warp_img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow("display", warp_img)
                cv2.waitKey(0)
                # Checking if the warp has 81 boxes
                print("Checking 1")
                if (check(warp_img)):
                    print("Correct 1")
                    # getting 81 boxes
                    gray_warp_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
                    rois = split_boxes(gray_warp_img)
                    rois = np.array(rois).reshape(-1,
                                                  input_size, input_size, 1)

                    # predicting wrap image
                    predicted_numbers = predict_sudoku_boxes(rois)
                    board_num = np.array(predicted_numbers).astype(
                        'uint8').reshape(9, 9)

                    # solving the sudoku

                    try:
                        solved_board_nums = solver.get_board(board_num)
                        # mask
                        binArr = np.where(
                            np.array(predicted_numbers) > 0, 0, 1)
                        flat_solved_board_nums = solved_board_nums.flatten()*binArr

                        # displaying text
                        text_img = display_numbers(
                            warp_img, flat_solved_board_nums)
                        # cv2.imshow("text shoow", text_img)

                        # Inverse warp Transform
                        inv = get_InvPerspective(img, text_img, approx)

                        combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
                        return contours, text_img, combined
                    except:
                        print("Sudoku Cannot be solved! VALUE ERROR 1")
                        pass

            warp_img = cv2.flip(warp_img, 1)
            for i in range(4):
                cv2.imshow("display", warp_img)
                cv2.waitKey(0)
                warp_img = cv2.rotate(warp_img, cv2.ROTATE_90_CLOCKWISE)
                # Checking if the warp has 81 boxes
                print("Checking 2")
                if (check(warp_img)):
                    print("Correct 2")
                    # getting 81 boxes
                    gray_warp_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
                    rois = split_boxes(gray_warp_img)
                    rois = np.array(rois).reshape(-1,
                                                  input_size, input_size, 1)

                    # predicting wrap image
                    predicted_numbers = predict_sudoku_boxes(rois)
                    board_num = np.array(predicted_numbers).astype(
                        'uint8').reshape(9, 9)

                    # solving the sudoku

                    try:
                        solved_board_nums = solver.get_board(board_num)
                        # mask
                        binArr = np.where(
                            np.array(predicted_numbers) > 0, 0, 1)
                        flat_solved_board_nums = solved_board_nums.flatten()*binArr

                        # displaying text
                        text_img = display_numbers(
                            warp_img, flat_solved_board_nums)
                        # cv2.imshow("text shoow", text_img)

                        # Inverse warp Transform
                        inv = get_InvPerspective(img, text_img, approx)

                        combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
                        return contours, text_img, combined
                    except:
                        print("Sudoku Cannot be solved! VALUE ERROR 2")
                        pass
    return contours, None, None


# vid = cv2.VideoCapture(0)
# ans = [[]]
# while (True):
#     ret, img = vid.read()
#     cv2.imshow('img', img)

#     contours, solved_warp_img, inv_solved_warp_img = solveSudoku(img)

#     cv2.imshow("Testing ", cv2.drawContours(
#         img, contours, -1, (0, 255, 0), 3))

#     if (solved_warp_img is not None and inv_solved_warp_img is not None):
#         ans = inv_solved_warp_img
#         # cv2.imshow("solved_warp", solved_warp_img)
#         # cv2.imshow("img", inv_solved_warp_img)
#         break
#     else:
#         print("No Sudoku Detected!")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# cv2.destroyAllWindows()

# cv2.imshow("Sudoku Solved", ans)
# cv2.waitKey(0)


# img = cv2.imread('photo.jpg')
# cv2.imshow('img', img)

# contours, solved_warp_img, inv_solved_warp_img = solveSudoku(img)

# cv2.imshow("Testing ", cv2.drawContours(
#     img, contours, -1, (0, 255, 0), 3))

# if (solved_warp_img is not None and inv_solved_warp_img is not None):
#     cv2.imshow("solved_warp", solved_warp_img)
#     cv2.imshow("inv_solved_wrap", inv_solved_warp_img)
# else:
#     print("No Sudoku Detected!")


# cv2.waitKey(0)
# cv2.destroyAllWindows()

model = tensorflow.keras.models.load_model('ocr_model.h5')
line_min_width = 20


img = cv2.imread('photo2.jpg')
cv2.imshow('img', img)

# thres = 127
# n, m = gray.shape
# for i in range(n):
#     for j in range(m):
#         if (gray[i][j] > thres):
#             gray[i][j] = 255
#         else:
#             gray[i][j] = 0
# cv2.imshow('im g', gray)

contours, solved_warp_img, inv_solved_warp_img = solveSudoku(img)

cv2.imshow("Testing ", cv2.drawContours(
    img, contours, -1, (0, 255, 0), 3))

if (solved_warp_img is not None and inv_solved_warp_img is not None):
    cv2.imshow("solved_warp", solved_warp_img)
    cv2.imshow("inv_solved_wrap", inv_solved_warp_img)
else:
    print("No Sudoku Detected!")


cv2.waitKey(0)
cv2.destroyAllWindows()
