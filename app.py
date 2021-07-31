import streamlit as st
from centernet.appmodule import explore_dataset
from centernet.commons import read_camera_intrinsic


def run_app():
    option = st.sidebar.selectbox(
        'Please select an option:',
        ('', 'Explore Dataset')
    )
    if option == 'Explore Dataset':
        explore_dataset('./data/pku-autonomous-driving/')
    read_camera_intrinsic('./data/pku-autonomous-driving/')


if __name__ == '__main__':
    run_app()
