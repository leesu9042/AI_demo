# views/data_page.py
import streamlit as st
import pandas as pd
import os


def show_data_info(df, title):
    st.subheader(title)

    st.write("데이터 미리보기")
    st.dataframe(df.head())

    row_count = df.shape[0]
    col_count = df.shape[1]
    missing_count = df.isnull().sum().sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("행 개수", row_count)
    col2.metric("컬럼 개수", col_count)
    col3.metric("전체 결측치", missing_count)

    with st.expander("컬럼별 결측치 보기"):
        missing_df = pd.DataFrame({
            "컬럼명": df.columns,
            "결측치 개수": df.isnull().sum().values
        })

        st.dataframe(missing_df)




def show_data_page():
    st.title("CSV Loader")

    st.divider()

    st.subheader("1. CSV 파일 업로드")

    labeled_upload = st.file_uploader(
        "라벨 있는 CSV 업로드",
        type=["csv"],
        key="labeled_upload"
    )


    if labeled_upload is not None:
        labeled_df = pd.read_csv(labeled_upload)
        st.session_state.labeled_df = labeled_df
        show_data_info(labeled_df, "라벨 있는 데이터")


    st.divider()

    st.subheader("2. CSV 경로로 불러오기")

    labeled_path = st.text_input("라벨 있는 CSV 경로 입력")

    if st.button("데이터 불러오기"):
        if labeled_path:
            if os.path.exists(labeled_path):
                labeled_df = pd.read_csv(labeled_path, nrows=1000)
                st.session_state.labeled_df = labeled_df
                show_data_info(labeled_df, "라벨 있는 데이터")
            else:
                st.error("라벨 있는 데이터 경로가 존재하지 않습니다.")



    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("학습하기"):
            if "labeled_df" in st.session_state:
                st.session_state.page = "learn"
                st.rerun()
            else:
                st.error("먼저 라벨 있는 데이터를 불러와주세요.")

    with col2:
        if st.button("홈으로 이동"):
            st.session_state.page = "home"
            st.rerun()