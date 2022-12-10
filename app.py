import cv2
from sudokudns import *
import streamlit as st

st.set_page_config(page_title='AR Sudoku Solver')
st.title("Augumented Reality Sudoku Solver")
image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
if image_file is not None:
    resetSudoku()
    st.image(image_file, width=250)
    loc = "temp"
    with open(loc, "wb") as f:
        f.write(image_file.getbuffer())
    st.success("Saved File")
    img = cv2.imread(loc)
    st.image(img, width=500, caption="Input Sudoku")
    solvedImage = solveSudoku(img)
    if (solvedImage is not None):
        st.image(solvedImage, width=500, caption="Perspective Transform")
    else:
        st.write("No Sudoku Found!")
