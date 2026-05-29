from io import StringIO

import pandas as pd
import streamlit as st

from services.prediction_service import prepare_prediction_input


def parse_excel_paste_text(paste_text):
    if not paste_text or not paste_text.strip():
        return None, ["붙여넣은 데이터가 없습니다."]

    try:
        pasted_df = pd.read_csv(
            StringIO(paste_text.strip()),
            sep="\t",
            dtype=str,
        )
    except Exception as e:
        return None, [f"붙여넣은 데이터를 읽을 수 없습니다: {e}"]

    if pasted_df.empty:
        return None, ["붙여넣은 데이터에 행이 없습니다."]

    if len(pasted_df) != 1:
        return None, ["컬럼명 포함 데이터 1행만 붙여넣어주세요."]

    pasted_df = pasted_df.fillna("")
    raw_input = pasted_df.iloc[0].to_dict()

    return raw_input, []


def validate_required_columns(raw_input, schema):
    required_cols = schema.get("required_input_cols", [])

    missing_cols = [
        col for col in required_cols
        if col not in raw_input
    ]

    if missing_cols:
        return [f"필수 컬럼이 없습니다: {missing_cols}"]

    return []


def show_prediction_input_page():
    st.title("예측 데이터 입력")

    selected_model = st.session_state.get("selected_model_metadata")

    if selected_model is None:
        st.error("선택된 모델이 없습니다.")
        if st.button("모델 선택으로 이동"):
            st.session_state.page = "prediction_model_select"
            st.rerun()
        return

    schema = selected_model.get("predict_input_schema")

    if not schema:
        st.error("예측 입력 스키마가 없습니다.")
        return

    required_cols = schema.get("required_input_cols", [])

    st.caption("엑셀에서 컬럼명(header)과 데이터 1행을 함께 복사해서 붙여넣어주세요.")
    st.dataframe(
        pd.DataFrame([required_cols], index=["필요 컬럼"]),
        use_container_width=True,
    )

    paste_text = st.text_area(
        "엑셀 데이터 붙여넣기",
        height=160,
        placeholder="EQUIP_NAME\tPART_NAME\tInjection_Time\t...\n650톤-우진2호기\tCN7 W/S SIDE MLD'G LH\t1.23\t...",
    )

    raw_input = None
    errors = []

    # 빈칸이 아니면 if문 실행(붙여넣은 데이터가 있으면 실행) 
    if paste_text.strip():
        raw_input, parse_errors = parse_excel_paste_text(paste_text)
        errors.extend(parse_errors)

        if raw_input is not None:
            errors.extend(validate_required_columns(raw_input, schema))

    result = {
        "is_valid": False,
        "errors": errors,
        "model_input_df": None,
    }

    # 입력데이터 에러 검
    if raw_input is not None and not errors:
        result = prepare_prediction_input(raw_input, schema)

    for error in result["errors"]:
        st.warning(error)

    if result["is_valid"]:
        st.success("입력 검증이 완료되었습니다.")

    if st.button(
        "다음",
        disabled=not result["is_valid"],
        use_container_width=True,
    ):
        st.session_state.predict_raw_input = raw_input
        st.session_state.predict_model_input_df = result["model_input_df"]
        st.session_state.page = "prediction"
        st.rerun()