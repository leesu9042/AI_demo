import pandas as pd
import logging

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,  accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ALLOWED_EQUIP_NAMES = ["650톤-우진2호기"]

PRODUCT_GROUP_MAPPING = {
    "CN7 W/S SIDE MLD'G LH": "CN7",
    "CN7 W/S SIDE MLD'G RH": "CN7",
    "RG3 MOLD'G W/SHLD, LH": "RG3",
    "RG3 MOLD'G W/SHLD, RH": "RG3",
}

DROP_COLS = [
    "_id",
    "TimeStamp",
    "PART_FACT_PLAN_DATE",
    "Reason",
    "PART_FACT_SERIAL",
    "PART_NAME",
    "EQUIP_CD",
    "EQUIP_NAME",
    "Mold_Temperature_1",
    "Mold_Temperature_2",
    "Mold_Temperature_5",
    "Mold_Temperature_6",
    "Mold_Temperature_7",
    "Mold_Temperature_8",
    "Mold_Temperature_9",
    "Mold_Temperature_10",
    "Mold_Temperature_11",
    "Mold_Temperature_12",
    "Barrel_Temperature_7",
]



def build_model_input(df, target_col=None):
    df = df.copy()

    df = df[df["EQUIP_NAME"].isin(ALLOWED_EQUIP_NAMES)].copy()
    df["PART_GROUP"] = df["PART_NAME"].map(PRODUCT_GROUP_MAPPING)
    df = df[df["PART_GROUP"].notna()].copy()

    if target_col is not None and target_col in df.columns:
        df[target_col] = df[target_col].map({
            "Y": 0,
            "N": 1
        })

    df.drop(DROP_COLS, axis=1, inplace=True, errors="ignore")

    return df





def train_model(df, target_col=None, test_size=0.2, random_state=42):
     # 1. 전처리 (1번전처리)
    df = build_model_input(df, target_col=target_col)

    # 2. X, y 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. train/test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. 컬럼 구분
    categorical_cols = ["PART_GROUP"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    required_input_cols = ["EQUIP_NAME", "PART_NAME"] + numeric_cols




    # 5. Pipeline 전처리 (2번전처리 - 원핫인코딩)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # 6. 모델 Pipeline
    model = Pipeline([
        ("preprocess", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # 7. 학습
    model.fit(X_train, y_train)

    feature_names = ( #컬럼 이름 복원
        preprocess.named_transformers_['cat']
        .get_feature_names_out(categorical_cols)
        .tolist()
        + numeric_cols
    )
    # 
    feature_importance: DataFrame = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.named_steps["rf"].feature_importances_ #모델이 각 피처에 부여한 중요도
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # 8. 예측 및 평가
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    cm = confusion_matrix(y_test, pred)
    
    
    logger.info(f"y_test 개수: {len(y_test)}")
    logger.info(f"혼동 행렬:\n{cm}")



    pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, pred_proba)


        
    return {
    "model": model, #pipeline 전체 반환
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    },
    "confusion_matrix": cm,
    "feature_names": feature_names,
    "feature_importance": feature_importance, # feature importance 해석용.
    
    # 예측 화면 검증용
    "predict_input_schema":  {
        "required_input_cols": required_input_cols,


        "raw_categorical_cols": {
            "EQUIP_NAME": ALLOWED_EQUIP_NAMES,
            "PART_NAME": list(PRODUCT_GROUP_MAPPING.keys()),
        },

        "allowed_values": {
            "EQUIP_NAME": ALLOWED_EQUIP_NAMES,
            "PART_NAME": list(PRODUCT_GROUP_MAPPING.keys()),
        },
        "mappings": {
            "PART_NAME": {
                "target_col": "PART_GROUP",
                "values": PRODUCT_GROUP_MAPPING,
            }
        },
        "model_input_cols": X.columns.tolist(),
        "categorical_cols": categorical_cols,
        "numeric_input_cols": numeric_cols,
    }




    


    }


