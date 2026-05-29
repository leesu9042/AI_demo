import pandas as pd

from train import build_model_input


# raw 1개 카테고리컬 컬럼(문자열)에 대한 허용된 범주값 검증 함수
def validate_raw_categories(raw_input, schema):
    errors = []

    raw_categorical_cols = schema.get("raw_categorical_cols", {})

    for col, allowed_values in raw_categorical_cols.items():
        value = raw_input.get(col)

        if value in (None, ""):
            errors.append(f"{col} 값이 없습니다.")
            continue

        if value not in allowed_values:
            errors.append(f"{col} 값 '{value}'은 허용된 범주에 없습니다.")

    return errors


# 전처리 후 모델 입력값 검증 함수
def validate_model_input(model_input_df, schema):
    errors = []

    model_input_cols = schema.get("model_input_cols", [])
    numeric_input_cols = schema.get("numeric_input_cols", [])

    if model_input_df is None or model_input_df.empty:
        errors.append("전처리 후 예측 가능한 데이터가 없습니다.")
        return errors

    missing_cols = [
        col for col in model_input_cols
        if col not in model_input_df.columns
    ]

    if missing_cols:
        errors.append(f"모델 입력 컬럼이 부족합니다: {missing_cols}")

    if missing_cols:
        return errors

    model_input_df = model_input_df[model_input_cols]

    if model_input_df.isnull().any().any():
        errors.append("모델 입력값에 비어있는 값이 있습니다.")

    for col in numeric_input_cols:
        if col not in model_input_df.columns:
            continue

        try:
            model_input_df[col].astype(float)
        except ValueError:
            errors.append(f"{col} 값은 숫자여야 합니다.")

    return errors

# 입력값 검증 흐름 
def prepare_prediction_input(raw_input, schema):

    # 1.raw input 검증
    raw_errors = validate_raw_categories(raw_input, schema)

    if raw_errors:
        return {
            "is_valid": False,
            "errors": raw_errors,
            "model_input_df": None,
        }
    # 2. 모델 입력값 생성 및 검증 (전처리)
    raw_input_df = pd.DataFrame([raw_input])
    model_input_df = build_model_input(raw_input_df)
    model_errors = validate_model_input(model_input_df, schema)


    if model_errors:
        return {
            "is_valid": False,
            "errors": model_errors,
            "model_input_df": None,
        }
    # 3. 검증 통과한 모델 입력값 반환
    model_input_cols = schema.get("model_input_cols", [])
    numeric_input_cols = schema.get("numeric_input_cols", [])

    # 컬럼 순서를 모델 학습 때와 똑같이 맞추는 코드
    model_input_df = model_input_df[model_input_cols].copy()


    # 사용자 입력은 보통 문자열로 들어오기 때문에 숫자형 컬럼은 float으로 변환
    for col in numeric_input_cols:
        if col in model_input_df.columns:
            model_input_df[col] = model_input_df[col].astype(float)

    return {
        "is_valid": True,
        "errors": [],
        "model_input_df": model_input_df,
    }
