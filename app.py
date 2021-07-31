import streamlit as st
from centernet.appmodule.commons import add_heading
from centernet.appmodule import explore_dataset, data_loader_module


def run_app():
    add_heading(
        content='Peking University/Baidu - Autonomous Driving Dataset',
        heading_level=1, align_center=True, add_hr=True
    )
    option = st.sidebar.selectbox(
        'Please select an option:',
        ('', 'Explore Dataset', 'Data Loader')
    )
    if option == 'Explore Dataset':
        explore_dataset('./data/pku-autonomous-driving/')
    elif option == 'Data Loader':
        data_loader_module('./data/pku-autonomous-driving/')


if __name__ == '__main__':
    run_app()
