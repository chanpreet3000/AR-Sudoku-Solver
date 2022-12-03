import streamlit as st
import cv2
from newdetect5 import *
introduction = str("One of the logic puzzles is sudoku, where the goal is to fill a 9x9 grid with numbers so that each row and column has all digits from 1 to 9. They have presumed that the greatest contour in the frame represents the sudoku puzzle to be solved in the conventional system. This may not always be the case, though. The suggested system determines the largest contour to solve the aforementioned problem and divides the contour into 81 pieces, with each piece standing in for a cell in a Sudoku puzzle, which has 9 x 9 cells. Once the collected image has undergone the necessary pre-processing, an efficient corner extraction technique is employed to identify the grid. A template-matching algorithm for each cell locates the digit utilizing the template.")
st.set_page_config(page_title='AR Sudoku Solver')
st.title("Augmented Reality Sudoku Solver")
st.subheader("Introduction.")
st.write(introduction)
st.subheader("Feed me Sudoku.")


run = st.checkbox('RESET SUDOKU')
# FRAME_WINDOW = st.image([])
OUTPUT_WINDOW = st.image([])
camera = cv2.VideoCapture(0)


while run:
    _, frame = camera.read()
    if(frame is not None):
        output = solveSudoku(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # FRAME_WINDOW.image(frame)
        OUTPUT_WINDOW.image(output)

else:
    st.write('Sudoku Reset')
    resetSudoku()
