import pandas as pd

from train import build_model_input


# 입력값 검증 흐름 
def prepare_prediction_input(raw_input_df, schema):


    # 1. df 필수컬럼 검증
    raw_errors = validate_raw_required_columns(raw_input_df, schema)

    if raw_errors:
        return {
            "is_valid": False,
            "errors": raw_errors,
            "model_input_df": None,
        }


    # 2. df 필수컬럼 값 유효 검증
    raw_errors = validate_raw_required_values(raw_input_df, schema)

    if raw_errors:
        return {
            "is_valid": False,
            "errors": raw_errors,
            "model_input_df": None,
        }

    # 3.df 카테고리 컬럼 검증
    raw_errors = validate_raw_categories(raw_input_df, schema)

    if raw_errors:
        return {
            "is_valid": False,
            "errors": raw_errors,
            "model_input_df": None,
        }
    


    # 4. 모델 학습 입력값 생성 및 검증 (전처리)
    model_input_df = build_model_input(raw_input_df)
    model_errors = validate_model_input(model_input_df, schema)


    if model_errors:
        return {
            "is_valid": False,
            "errors": model_errors,
            "model_input_df": None,
        }
    
    # 5. 검증 통과한 모델 입력값 반환
    pipeline_input_cols = schema.get("pipeline_input_cols", [])
    numeric_input_cols = schema.get("numeric_input_cols", [])

    # 컬럼 순서를 모델 학습 때와 똑같이 맞추는 코드
    model_input_df = model_input_df[pipeline_input_cols].copy()


    # 사용자 입력은 보통 문자열로 들어오기 때문에 숫자형 컬럼은 float으로 변환
    for col in numeric_input_cols:
        if col in model_input_df.columns:
            model_input_df[col] = model_input_df[col].astype(float)

    return {
        "is_valid": True,
        "errors": [],
        "model_input_df": model_input_df,
    }




# ----- 검증 모듈함수들 
# ----  전처리전 입력값 검증 함수

# raw 필수컬럼 검증 함수
def validate_raw_required_columns(df, schema):
    errors = []
    required_cols = schema.get("raw_required_cols", [])

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        errors.append(f"필수 컬럼이 없습니다: {missing_cols}")

    return errors

# 필수컬럼 값 들어있는지 검증 함수
def validate_raw_required_values(df, schema):
    required_cols = schema.get("raw_required_cols", [])

    errors = []

    for col in required_cols:
        if col not in df.columns:
            continue

        invalid_mask = (
            df[col].isna()
            | (df[col].astype(str).str.strip() == "")
        )

        if invalid_mask.any(): #하나라도 true면 true 반환
            errors.append(
                f"{col} 컬럼에 빈 값이 있습니다."
            )

    return errors



# raw 1개 카테고리컬 컬럼(문자열)에 대한 허용된 범주값 검증 함수
def validate_raw_categories(df, schema):
    errors = []

    allowed_values_map = schema.get("allowed_values", {})

    for col, allowed_values in allowed_values_map.items():

        if col not in df.columns:
            continue
        # invalid_values = 실제값 - 허용값
        invalid_values = (
            set(df[col].dropna())
            - set(allowed_values)
        )

        if invalid_values: # 빈 set은 false, 하나라도 있으면 true 반환
            errors.append(
                f"{col} 컬럼에 허용되지 않은 값이 있습니다: {list(invalid_values)}"
            )

    return errors



#--------- 전처리후

# 전처리 후 모델 입력값 유효 검증 함수

def validate_model_input(model_input_df):
    errors = []

    if model_input_df is None or model_input_df.empty:
        errors.append("전처리 후 예측 가능한 데이터가 없습니다.")

    return errors