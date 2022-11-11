from solver import *
import streamlit as st
import cv2
from sudoku_ip import *

st.set_page_config(page_title='AR Sudoku Solver')
st.title("Augumented Reality Sudoku Solver")
st.subheader("Feed me Sudoku.")

image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
if image_file is not None:
    st.image(image_file,width=250)
    loc = "photo/temp"
    with open(loc,"wb") as f: 
      f.write(image_file.getbuffer())      
    st.success("Saved File")
    img = cv2.imread(loc)
    st.image(img, width=500, caption="Input Sudoku")
    wrapImage, solvedImage  = solveTheSudoku(img)
    if(wrapImage is not None and solvedImage is not None):
        st.image(wrapImage, width=500, caption="Perspective Transform")
        st.image(solvedImage, width=500, caption="Solved Sudoku!")
    else:
        st.write("No Sudoku Found!")
