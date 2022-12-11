import cv2

import numpy as np

import tensorflow

import solver##imported file




model = tensorflow.keras.models.load_model('ocr_model.h5')##model is loaded the ocr model is loaded

height, width = 630, 630
##height and width will be adjusted because 630/9 = 9 and thus every box will have  70 pixels (9 box ,70 pixels)
input_size = 48
##input size  70*70 image is received and then given to 48*48 for our model to work on it(ocr)


def get_InvPerspective(img, masked_num, location):##original image , solved masked num(black image),location is also sent
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])##wblack image corners are sent
    pts2 = np.float32([location[0], location[3], location[1], location[2]])##warp image corners are sent 
    ##this is done to make both the warped image and as well as the black image coincide so as so to superimpose the image

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(
        masked_num, matrix, (img.shape[1], img.shape[0]))##this is applied so as to get the original image without warping
    return result


def display_numbers(img, numbers, color=(0, 0, 255)):##solved numbers will be sent here
    W = int(height/9)##width and height is taken for to fit in box
    H = int(width/9)
    for i in range(9):##in every box
        for j in range(9):
            if numbers[(j*9)+i] != 0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int(##puttext done = text is printed (image is sent)
                ##numbers will be put on black image and green color is taken where the numbers were not 0 aka where it has to printed
                    (j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img##image returned


def predict_sudoku_boxes(rois):
    prediction = model.predict(rois)##model (ocr) we will predict the number whats written
    predicted_numbers = []##prediction array
    for i in prediction:
        index = (np.argmax(i))##prediction is done and thus we will have a probability of each number (1-9)and thus the highest convoluted 
        ##will be choosen out of all predicted values
        predicted_number = index##array append
        predicted_numbers.append(predicted_number)##81 numbers will be predicted

    return predicted_numbers##size = 81


def split_boxes(board):##630*630 is converted into 9*9 
    rows = np.vsplit(board, 9)##we will have an array of size 9[here 9*9 boxes will be converted for the number]
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0##resize into 48*48 so that it can be recognized by ocr model
            boxes.append(box)
    return boxes


def get_predicted_board(img):
    gray_warp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)##gray level image
    rois = split_boxes(gray_warp_img)##rows image got it
    rois = np.array(rois).reshape(-1,
                                  input_size, input_size, 1)##again 48*48

    # predicting wrap image
    predicted_numbers = predict_sudoku_boxes(rois)##
    board_num = np.array(predicted_numbers).astype(
        'uint8').reshape(9, 9)##and that 81 size matrix converted into 9*9
    print("Predicted Numbers : ", predicted_numbers)
    return board_num##board num is achieved


def solve_warped_board(board_num):
    try:
        return solver.get_board(board_num.copy())##solver .get_board we willl send a copy of the numbers
    except:
        return []##else we will send a normal array(empty)

##checking if there are 81 boxes or not
def check_sudoku(img):
    contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)##contours also found again to check the 9*9 boxes

    num = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 15, True)
        if len(approx) == 4:##square found again and thus nums is increased
            num += 1
    # print(num)
    if (num >= 10):##threshold taken so that it detects that it is sudoko
        return True
    else:
        return False


def get_perspective(img, location):##array location of contours with image
    pts1 = np.float32([location[0], location[3], location[1], location[2]])##it will take all the four corners
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])##and zoom it till these 4 coordinates (48*48)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)##cvfunction pts1 and pts2 will return perspective transform
    ##matrix will be given in warp image(perspective transform)
    result = cv2.warpPerspective(img, matrix, (width, height)) ##warp perspective is done
    return result##result


def solve(frame):
    img = frame.copy() ##frame hardcopy in image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)##gray(black and white)(gray level image)
    gray = cv2.GaussianBlur(gray, (3, 3), 20)##gaussianblue is used to remove noise and blurr 
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)##mean thresholding is used because to have a 
                                   ##better thresholding and boundaries are easily distinguished 

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)##contours  = we gave threshold values and and thus when  the paramters are passed we
        ##get boundaries ,and here we will take the maximum of 20 contours

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]##calculate the area of each contours sort them by their areas and 
    ##therefore we will select only the top 20 for boundary of the sudoku

    for cnt in contours:
        cnt_position = cv2.approxPolyDP(cnt, 15, True)##traversing in contours,approxPolydp will find the edges(corners)(4coordinates are there)

        if len(cnt_position) == 4:##if corners = 4 square
            
            warp_img = get_perspective(img, cnt_position)##warp image = perspective transform(image will be zoomed in and we will only be left with sudoko)
            ##other things in image will be cleared
            warp_thesh_img = get_perspective(thresh, cnt_position)##warp thres image (one done on normal image and another is on thres image for letter detection)


            if (check_sudoku(warp_thesh_img)):##warp image bhejo
                predicted_board = get_predicted_board(warp_img)##predicted board size = 81
                solved_board = solve_warped_board(predicted_board) ##solve willbe done by sending predicted number
                print("Predicted Board : ", predicted_board)
                if (len(solved_board) != 0):##is length of solved board is not 0 
                    binArr = np.where(
                        np.array(predicted_board.flatten()) > 0, 0, 1)##binary array (green text shown only ) ##make an array where the
                        ##predicted  numbers present in board will be 0 and empty box will be 1(9*9 flattened = 81)
                    temp = np.array(
                        solved_board.copy()).flatten()##binary mask where 0 that text will not be printed and other text will be printed
                    flatten_solved_board = temp*binArr##multiplied by binary masking
                    solved_sudoku_board = flatten_solved_board.copy()##flattened array should not changed so made a copy
                    print("Solved Board : ", solved_sudoku_board)

                    black = np.zeros((height, width, 3), dtype="uint8")##black color image (intensity = 0)(black image)

                    text_img = display_numbers(##paramters solved sudoko board and black image taken
                        black, solved_sudoku_board, (0, 255, 0))##color is defined (bgr value = green image)

                    inv = get_InvPerspective(img, text_img, cnt_position)##inverse perspective is called

                    temp = img.copy()
                    combined = cv2.addWeighted(temp, 0.5, inv, 0.5, 0)##addweighted is used to superimpose the black image and warped image
                    ##weights are same so as to get the image
                    return combined##combined is sent

    return []



def solveSudoku(img):
    if (len(img) != 0):
        combined = solve(img)## if image not present return null
        if (len(combined) != 0):##solve the image
            return combined
    return img##solved sudoku solved under combined and thus image is returned


# img = cv2.imread('photo2.jpg')
# output = solveSudoku(img)
# cv2.imshow("Output", output)
# cv2.waitKey(0)

# vid = cv2.VideoCapture(0)
# while (True):
#     ret, frame = vid.read()
#     cv2.imshow("Input", frame)
#     combined = solveSudoku(frame)
#     cv2.imshow("Output", frame)
#     if (len(combined) != 0):
#         cv2.imshow("Output", combined)
#     # resetSudoku()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# cv2.destroyAllWindows()
