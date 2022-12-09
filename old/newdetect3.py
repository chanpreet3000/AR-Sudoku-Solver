import cv2
import numpy as np
import tensorflow
import solver


model = tensorflow.keras.models.load_model('ocr_model.h5')
height, width = 630, 630
input_size = 48


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


def predict_sudoku_boxes(rois):
    prediction = model.predict(rois)
    predicted_numbers = []
    for i in prediction:
        index = (np.argmax(i))
        predicted_number = index
        predicted_numbers.append(predicted_number)

    return predicted_numbers


def split_boxes(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    return boxes


def solve_warped_board(img):
    gray_warp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray_warp_img)
    rois = np.array(rois).reshape(-1,
                                  input_size, input_size, 1)

    # predicting wrap image
    predicted_numbers = predict_sudoku_boxes(rois)
    board_num = np.array(predicted_numbers).astype(
        'uint8').reshape(9, 9)

    try:
        solved_board_nums = solver.get_board(board_num)
        print(solved_board_nums)

        # mask
        binArr = np.where(
            np.array(predicted_numbers) > 0, 0, 1)
        flat_solved_board_nums = solved_board_nums.flatten()*binArr
        return flat_solved_board_nums
    except:
        return []


def check_sudoku(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))

    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    num = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 15, True)
        if len(approx) == 4:
            num += 1
    # print(num)
    if (num >= 100):
        return True
    else:
        return False


def get_perspective(img, location):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def get_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))

    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def solve(frame):
    img = frame.copy()
    img = cv2.resize(img, (height, width))
    # cv2.imshow('Image', img)

    contours = get_contour(img)

    for cnt in contours:
        cnt_position = cv2.approxPolyDP(cnt, 15, True)
        if len(cnt_position) == 4:
            cnt_img = img.copy()
            cv2.drawContours(cnt_img, cnt, -1, (0, 255, 0), 3)
            cv2.imshow("Contour Image", cnt_img)

            warp_img = get_perspective(img, cnt_position)

            if (check_sudoku(warp_img)):
                cv2.imshow("Warp Image", warp_img)

                flat_solved_board_nums = solve_warped_board(warp_img)

                if (len(flat_solved_board_nums) != 0):
                    black = np.zeros((height, width, 3), dtype="uint8")

                    text_img = display_numbers(
                        black, flat_solved_board_nums, (0, 255, 0))

                    # cv2.imshow("text shoow", text_img)

                    # Inverse warp Transform
                    inv = get_InvPerspective(img, text_img, cnt_position)

                    temp = img.copy()
                    combined = cv2.addWeighted(temp, 1, inv, 1, 0)
                    return combined

            return []
    return []


# frame = cv2.imread('photo2.jpg')
# solve(frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()
    combined = solve(frame)
    if (len(combined) != 0):
        cv2.imshow("Inverse Perspective", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
