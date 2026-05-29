import streamlit as st
from views.components.model_selector import show_model_selector


def _select_model(model_metadata):
    st.session_state.selected_model_metadata = model_metadata


def show_prediction_model_page():
    st.title("모델 예측")

    st.subheader("모델 선택")
    selected_model = show_model_selector(
        button_label="이 모델 선택",
        on_click=_select_model
    )

    if selected_model is None:
        selected_model = st.session_state.get("selected_model_metadata")

    if selected_model is not None:
        st.success(f"선택된 모델: {selected_model.get('model_name', '이름 없음')}")

        if st.button("모델 예측 진행"):
            st.session_state.page = "prediction_input"
            st.rerun()
            


    if st.button("홈으로 이동"):
        st.session_state.page = "home"
        st.rerun()
