import streamlit as st
from typing import Tuple
from matplotlib import pyplot as plt


def add_heading(content: str, heading_level: int, align_center: bool, add_hr: bool):
    html = '''<h{}>{}</h{}>'''.format(heading_level, content, heading_level)
    html = '<center>' + html + '</center>' if align_center else html
    st.markdown(html, unsafe_allow_html=True)
    if add_hr:
        st.markdown('<hr>', unsafe_allow_html=True)


def plot_image_matplotlib(image, title: str, figure_size: Tuple[int, int] = (18, 18)):
    plt.figure(figsize=figure_size)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)
