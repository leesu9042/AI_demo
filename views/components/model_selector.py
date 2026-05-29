# components/model_selector.py

import streamlit as st
from services.model_service import get_model_metadata_list


def _format_saved_at(saved_at):
    if not saved_at:
        return "저장일 없음"

    if len(saved_at) == 15 and saved_at[8] == "_":
        return (
            f"{saved_at[:4]}-{saved_at[4:6]}-{saved_at[6:8]} "
            f"{saved_at[9:11]}:{saved_at[11:13]}:{saved_at[13:15]}"
        )

    return saved_at


def _format_metric(value):
    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return str(value)


def _toggle_detail(detail_key):
    st.session_state[detail_key] = not st.session_state.get(detail_key, False)


def _get_summary_metric_items(metrics):
    metric_items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1", metrics.get("f1")),
        ("ROC-AUC", metrics.get("roc_auc")),
    ]

    return [
        (name, value)
        for name, value in metric_items
        if value is not None
    ]


def _show_metric_items(metric_items):
    if not metric_items:
        st.caption("저장된 성능 지표가 없습니다.")
        return

    metric_cols = st.columns(min(len(metric_items), 5))

    for col, (name, value) in zip(metric_cols, metric_items):
        col.metric(name, _format_metric(value))


def _show_detail(meta, description, saved_at):
    st.markdown(f"**설명**  \n{description}")
    st.markdown(f"**저장일시**  \n{saved_at}")

    st.divider()
    st.subheader("상세 데이터")

    metrics = meta.get("metrics", {})
    _show_metric_items(list(metrics.items()))

    st.markdown(f"**sklearn 버전**  \n{meta.get('sklearn_version', '')}")


def show_model_selector(
    button_label="선택",
    on_click=None,
    show_select_button=True,
    show_detail_button=False,
    show_summary_metrics=True,
    key_prefix="model_selector",
    models=None
):

    if models is None:
        models = get_model_metadata_list()

    if not models:
        st.info("저장된 모델이 없습니다.")
        return

    selected_model = None

    for idx, meta in enumerate(models):

        model_name = meta.get("model_name", "이름 없음")
        description = meta.get("description") or "설명 없음"

        saved_at = _format_saved_at(
            meta.get("saved_at", "")
        )

        metrics = meta.get("metrics", {})
        detail_key = f"{key_prefix}_detail_{idx}_{meta.get('saved_at', '')}"

        # 모델별 컨테이너 생성
        with st.container(border=True):

            action_count = int(show_select_button) + int(show_detail_button)
            if action_count:
                columns = st.columns([4] + [1] * action_count)
                header_col = columns[0]
                action_cols = columns[1:]
            else:
                header_col = st.container()
                action_cols = []

            with header_col:
                st.markdown(f"#### {model_name}")
                st.caption(f"저장일: {saved_at}")

            action_col_index = 0

            if show_select_button:
                with action_cols[action_col_index]:
                    if st.button(
                        button_label,
                        key=f"{key_prefix}_select_{idx}_{meta.get('saved_at', '')}",
                        use_container_width=True
                    ):
                        selected_model = meta

                action_col_index += 1

            if show_detail_button:
                with action_cols[action_col_index]:
                    detail_label = (
                        "닫기"
                        if st.session_state.get(detail_key, False)
                        else "자세히 보기"
                    )

                    st.button(
                        detail_label,
                        key=f"{key_prefix}_toggle_{idx}_{meta.get('saved_at', '')}",
                        on_click=_toggle_detail,
                        args=(detail_key,),
                        use_container_width=True
                    )

            if description:
                st.write(description)

            if show_summary_metrics:
                _show_metric_items(_get_summary_metric_items(metrics))

            if show_detail_button and st.session_state.get(detail_key, False):
                st.divider()
                _show_detail(meta, description, saved_at)

    if (
        show_select_button
        and selected_model is not None
        and on_click is not None
    ):
        on_click(selected_model)

    return selected_model
