import streamlit as st


def add_heading(content: str, heading_level: int, align_center: bool, add_hr: bool):
    html = '''<h{}>{}</h{}>'''.format(heading_level, content, heading_level)
    html = '<center>' + html + '</center>' if align_center else html
    st.markdown(html, unsafe_allow_html=True)
    if add_hr:
        st.markdown('<hr>', unsafe_allow_html=True)
