# views/learn_page.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train import train_model
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def show_learn_page():
    st.title(" 모델 학습 페이지")

    if "labeled_df" not in st.session_state:
        st.error("데이터가 없습니다. 먼저 데이터를 불러와주세요.")
        if st.button("데이터 페이지로 돌아가기"):
            st.session_state.page = "data"
            st.rerun()
        return

    labeled_df = st.session_state.labeled_df
    
    st.subheader(" 학습 데이터")
    st.dataframe(labeled_df.head(10))

    st.subheader(" 학습 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox(
            "타겟 컬럼 선택",
            options=labeled_df.columns,
            index=len(labeled_df.columns) - 1
        )
    
    with col2:
        test_size = st.slider(
            "테스트 데이터 비율",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )

    if st.button(" 모델 학습 시작", use_container_width=True):
        with st.spinner("모델을 학습 중입니다..."):
            try:
                results = train_model(labeled_df, target_col=target_col, test_size=test_size)
                st.session_state.model_results = results
                st.success(" 모델 학습 완료!")
            except Exception as e:
                st.error(f" 오류 발생: {str(e)}")
                return


    # 모델 학습완료
    if "model_results" in st.session_state:
        results = st.session_state.model_results
        
        # 모델이 사용한 피처 목록
        st.subheader(" 모델에 사용된 피처 (Features)")
        with st.expander("사용된 컬럼 목록 보기", expanded=False):
            feature_df = pd.DataFrame({
                "순번": range(1, len(results['feature_names']) + 1),
                "피처명": results['feature_names']
            })
            st.dataframe(feature_df, use_container_width=True)
            st.info(f"총 {len(results['feature_names'])}개의 피처 사용됨")


    # 모델이 각 피처에 부여한 중요도
        if "feature_importance" in results:
            importance_df = results["feature_importance"].copy() #가져올땐 복사본으로 가져오기

            st.subheader("특성 중요도 (Feature Importance)")

            if importance_df.empty:
                st.info("표시할 특성 중요도 데이터가 없습니다.")
            else:
                top_n = st.slider(
                    "표시할 특성 수",
                    min_value=1,
                    max_value=len(importance_df),
                    value=min(15, len(importance_df)),
                    step=1
                )

                top_importance = importance_df.head(top_n)
                chart_df = top_importance.sort_values("importance")

                fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
                ax.barh(chart_df["feature"], chart_df["importance"])
                ax.set_xlabel("Importance")
                ax.set_ylabel("")
                ax.set_title(f"Top {top_n} Feature Importance")
                st.pyplot(fig)

                display_df = top_importance.copy()
                display_df.insert(0, "순위", range(1, len(display_df) + 1))
                display_df["importance"] = display_df["importance"].round(6)
                st.dataframe(display_df, use_container_width=True)
        
        st.subheader("모델 성능 지표")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("정확도 (Accuracy)", f"{results['metrics']['accuracy']:.4f}")
        with col2:
            st.metric("정밀도 (Precision)", f"{results['metrics']['precision']:.4f}")
        with col3:
            st.metric("재현율 (Recall)", f"{results['metrics']['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{results['metrics']['f1']:.4f}")
        
        if results['metrics']['roc_auc'] is not None:
            col5 = st.columns(1)[0]
            with col5:
                st.metric("ROC-AUC", f"{results['metrics']['roc_auc']:.4f}")
        



        cm = results["confusion_matrix"]

        st.subheader(" 혼동 행렬 (Confusion Matrix)")

        fig, ax = plt.subplots(figsize=(5, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Defect"]
        )

        disp.plot(ax=ax)

        st.pyplot(fig)
        

        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("데이터 페이지로 돌아가기"):
            st.session_state.page = "data"
            st.rerun()
    
    with col2:
        if st.button("모델 저장", disabled="model_results" not in st.session_state): # 모델저장안되어있으면 비활성화
            st.success(" 저장 페이지로 이동합니다.")
            st.session_state.page = "save_model"
            st.rerun()



    with col3:
        if st.button("홈으로 이동"):
            st.session_state.page = "home"
            st.rerun()


