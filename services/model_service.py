# services/model_service.py
import os
import json
from datetime import datetime
import re


# 학습된 모델(Pipeline) 저장
#        +
# 모델 설명(metadata) 저장

def save_model(model_pipeline, metadata):
    import joblib
    import sklearn

    metadata = metadata.copy()

    os.makedirs("saved_models/model_files", exist_ok=True)
    os.makedirs("saved_models/metadata", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #경로 생성
    model_name = metadata["model_name"]

    safe_model_name = make_safe_filename(model_name)

    model_path = os.path.join(
    "saved_models",
    "model_files",
    f"{safe_model_name}_{timestamp}.pkl"
)

 
    # 1.모델 저장
    joblib.dump(model_pipeline, model_path)

    # 2.metadata 저장
    metadata["saved_at"] = timestamp
    metadata["sklearn_version"] = sklearn.__version__
    metadata["model_file_path"] = model_path

    meta_path = os.path.join(
        "saved_models",
        "metadata",
        f"{safe_model_name}_{timestamp}_meta.json"
    )

    # 메타데이터 저장
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return model_path


def load_model(model_file_path):
    import joblib

    return joblib.load(model_file_path)


# 저장된 모델 목록 불러오기 (정확히는 메타데이터)
def get_model_metadata_list():
    metadata_dir = os.path.join("saved_models", "metadata")

    if not os.path.isdir(metadata_dir):
        return []

    metadata_list = []

    for file_name in os.listdir(metadata_dir):
        if not file_name.endswith("_meta.json"):
            continue

        meta_path = os.path.join(metadata_dir, file_name)

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        metadata_list.append(metadata)

    return sorted(
        metadata_list,
        key=lambda item: item.get("saved_at", ""),
        reverse=True
    )



def make_safe_filename(name):

    safe = re.sub(
        r'[^a-zA-Z0-9가-힣_]',
        '_',
        name
    )

    safe = re.sub(r'_+', '_', safe)

    safe = safe.strip('_')

    return safe.lower()
