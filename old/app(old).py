from solver import *
import streamlit as st
import cv2
from old.sudoku_ip import *

introduction = str("One of the logic puzzles is sudoku, where the goal is to fill a 9x9 grid with numbers so that each row and column has all digits from 1 to 9. They have presumed that the greatest contour in the frame represents the sudoku puzzle to be solved in the conventional system. This may not always be the case, though. The suggested system determines the largest contour to solve the aforementioned problem and divides the contour into 81 pieces, with each piece standing in for a cell in a Sudoku puzzle, which has 9 x 9 cells. Once the collected image has undergone the necessary pre-processing, an efficient corner extraction technique is employed to identify the grid. A template-matching algorithm for each cell locates the digit utilizing the template.")
st.set_page_config(page_title='AR Sudoku Solver')
st.title("Augmented Reality Sudoku Solver")
st.subheader("Introduction.")
st.write(introduction)
st.subheader("Feed me Sudoku.")

image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
if image_file is not None:
    st.image(image_file,width=250)
    loc = "photo/temp"
    with open(loc,"wb") as f: 
      f.write(image_file.getbuffer())      
    st.success("Saved File")

    img = cv2.imread(loc)
    img = cv2.resize(img, (512, 512))
    st.image(img, width=500, caption="Input Sudoku")

    wrapImage, solvedImage  = solveTheSudoku(img)
    if(wrapImage is not None and solvedImage is not None):
        st.image(wrapImage, width=500, caption="Perspective Transform")
        st.image(solvedImage, width=500, caption="Solved Sudoku!")
    else:
        st.write("No Sudoku Found!")