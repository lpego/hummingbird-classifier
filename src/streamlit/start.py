import streamlit as st

st.set_page_config(
    page_title="BioDetect",
)

st.write("# Welcome to BioDetect")

st.sidebar.success("Select a task above.")

st.markdown(
    """
    BioDetect is a Machine Learning Detection Model for hummingbirds by [Luca Pegoraro](https://scholar.google.co.uk/citations?user=w07Pg5EAAAAJ&hl=en) \n
    **Select your task from the sidebar**
    # 
    # 
    #
    # 
    ### Resources and references
    - User Interface made with [streamlit.io](https://streamlit.io)
    - Inspired by [DeepMeerkat](http://benweinstein.weebly.com/deepmeerkat.html)
    # 
    # 
    ##### If you have questions contact [Luca.Pegoraro@WSL.ch](mailto:Luca.Pegoraro@WSL.ch)
    """
)