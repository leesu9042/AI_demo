import streamlit as st
from views.data_page import show_data_page
from views.learn_page import show_learn_page
from views.model_list_page import show_model_list_page
from views.prediction_input_page import show_prediction_input_page
from views.prediction_result_page import show_prediction_page
from views.prediction_model_select_page import  show_prediction_model_page
from views.save_model_page import show_save_model_page


if "page" not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":
    st.title("AI 데모 페이지")

    if st.button("1.새 모델 학습"):
        st.session_state.page = "data"
        st.rerun()
        
    if st.button("2.저장된 모델 목록"):
        st.session_state.page = "model_list"
        st.rerun()

    if st.button("3.모델 예측"):
        st.session_state.page = "prediction_model_select"
        st.rerun()

elif st.session_state.page == "data":
    show_data_page()

elif st.session_state.page == "learn":
    show_learn_page()


elif st.session_state.page == "save_model":
    show_save_model_page()


elif st.session_state.page == "model_list":
    show_model_list_page()


elif st.session_state.page == "prediction_model_select":
    show_prediction_model_page()


elif st.session_state.page == "prediction_input":
    show_prediction_input_page()

elif st.session_state.page == "prediction":
    show_prediction_page()