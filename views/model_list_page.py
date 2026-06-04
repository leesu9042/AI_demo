import streamlit as st

from services.model_service import get_model_metadata_list
from views.components.model_selector import show_model_selector


def _go_predict_page(model_metadata):
    st.session_state.selected_model_metadata = model_metadata
    st.session_state.page = "prediction_model_select"
    st.rerun()


def show_model_list_page():
    st.title("저장된 모델 목록")

    metadata_list = get_model_metadata_list()

    if not metadata_list:
        st.info("저장된 모델 메타데이터가 없습니다.")
        if st.button("홈으로 이동"):
            st.session_state.page = "home"
            st.rerun()
        return

    st.caption("자세히 보기 버튼을 누르면 상세 정보를 확인할 수 있습니다.")

    show_model_selector(
        button_label="예측",
        on_click=_go_predict_page,
        show_select_button=True,
        show_detail_button=True,
        show_summary_metrics=False,
        key_prefix="model_list",
        models=metadata_list,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("새로고침", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("홈으로 이동", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
