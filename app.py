import cv2
from sudokudns import *
import streamlit as st

st.set_page_config(page_title='AR Sudoku Solver')##title is set(link title)
st.title("Augumented Reality Sudoku Solver")
image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])##upload the image 
if image_file is not None:
    
    st.image(image_file, width=250)##width is made as the width of 250; print the image
    loc = "temp"##to make it compatible with opnecv
    with open(loc, "wb") as f:##
        f.write(image_file.getbuffer())##temp file is created in local storage
    st.success("Saved File")
    img = cv2.imread(loc)##the temp file is read
    st.image(img, width=500, caption="Input Sudoku")
    solvedImage = solveSudoku(img)
    if (solvedImage is not None):
        st.image(solvedImage, width=500, caption="Perspective Transform")##bgr is printed in opencv and st uses rgb thats why some slight change
    else:
        st.write("No Sudoku Found!")
