import pandas as pd
import streamlit as st

from services.model_service import load_model


def _format_prediction_label(value):
    try:
        label = int(value)
    except (TypeError, ValueError):
        return value

    return "불량" if label == 1 else "정상"


def _build_prediction_result_df(model_input_df, predictions, probabilities=None):
    result_df = model_input_df.copy()
    result_df["prediction"] = predictions
    result_df["prediction_label"] = [
        _format_prediction_label(value)
        for value in predictions
    ]

    if probabilities is not None:
        result_df["defect_probability"] = probabilities

    return result_df


def show_prediction_page():
    st.title("예측 결과")

    selected_model = st.session_state.get("selected_model_metadata")

    if selected_model is None:
        st.error("선택된 모델이 없습니다.")
        if st.button("모델 선택으로 이동"):
            st.session_state.page = "prediction_model_select"
            st.rerun()
        return

    model_input_df = st.session_state.get("predict_model_input_df")

    if model_input_df is None:
        st.error("예측할 데이터가 없습니다. 먼저 예측 데이터를 입력해주세요.")
        if st.button("예측 데이터 입력으로 이동"):
            st.session_state.page = "prediction_input"
            st.rerun()
        return

    model_name = selected_model.get("model_name", "이름 없음")
    model_file_path = selected_model.get("model_file_path")

    st.subheader("선택된 모델")
    st.success(model_name)

    if model_file_path:
        st.caption(model_file_path)
    else:
        st.error("모델 파일 경로가 없습니다.")
        return

    st.subheader("예측 입력 데이터")
    st.dataframe(model_input_df, use_container_width=True)

    if st.button("예측 실행", use_container_width=True):
        try:
            model = load_model(model_file_path)
            predictions = model.predict(model_input_df)

            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(model_input_df)[:, 1]

            result_df = _build_prediction_result_df(
                model_input_df,
                predictions,
                probabilities,
            )

            st.session_state.prediction_result_df = result_df
            st.success("예측이 완료되었습니다.")

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")

    result_df = st.session_state.get("prediction_result_df")

    if isinstance(result_df, pd.DataFrame):
        st.subheader("예측 결과")
        st.dataframe(result_df, use_container_width=True)

        if len(result_df) == 1:
            label = result_df.iloc[0]["prediction_label"]
            st.metric("결과", label)

            if "defect_probability" in result_df.columns:
                st.metric(
                    "불량 확률",
                    f"{result_df.iloc[0]['defect_probability']:.4f}",
                )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("데이터 다시 입력", use_container_width=True):
            st.session_state.pop("prediction_result_df", None)
            st.session_state.page = "prediction_input"
            st.rerun()

    with col2:
        if st.button("다른 모델 선택", use_container_width=True):
            st.session_state.pop("selected_model_metadata", None)
            st.session_state.pop("predict_model_input_df", None)
            st.session_state.pop("prediction_result_df", None)
            st.session_state.page = "prediction_model_select"
            st.rerun()

    with col3:
        if st.button("홈으로 이동", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
