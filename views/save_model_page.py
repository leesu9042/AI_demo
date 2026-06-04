# views/save_model_page.py
import streamlit as st
from services.model_service import save_model


def show_save_model_page():
    st.title("모델 저장")

    if "model_results" not in st.session_state:
        st.error("저장할 학습 결과가 없습니다.")
        if st.button("학습 페이지로 이동"):
            st.session_state.page = "learn"
            st.rerun()
        return

    results = st.session_state.model_results

    model_name = st.text_input("모델 이름", value="rf_model")
    description = st.text_area("설명", value="")

    st.subheader("성능 지표")
    st.json(results["metrics"])

    if st.button("저장하기", use_container_width=True):
        if not model_name.strip():
            st.error("모델 이름을 입력해주세요.")
            return

        metadata = {
            "model_name": model_name,
            "description": description,
            "metrics": results["metrics"],
            "feature_names": results["feature_names"],
            "predict_input_schema": results["predict_input_schema"],

        }

        model_path = save_model(results["model"], metadata)
        st.success(f"모델이 저장되었습니다: {model_path}")

    if st.button("home으로 이동"):
        st.session_state.page = "home"
        st.rerun()


        