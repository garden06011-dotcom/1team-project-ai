# population_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# -----------------------------
# 1. 설정
# -----------------------------
DATA_PATH = "final-data.xlsx"           # 엑셀 파일 경로
MODEL_PATH = "population_rf_model.pkl"

TARGET_COLS = [
    "길단위유동인구",
    "주거인구",
    "전체점포수",
    "음식점점포수",
    "카페점포수",
    "호프점포수",
    "당월매출",
    "주말매출",
    "주중매출"
]

# -----------------------------
# 2. 데이터 로드 및 기본 정렬
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 기본 정렬: 행정동, 년도, 분기
    df = df.sort_values(["행정동", "년도", "분기"]).reset_index(drop=True)
    print("-------------- load_data ------------------")
    print(df.head())
    return df

# -----------------------------
# 3. 시계열 피처(lag, rolling) 생성
# -----------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # 년도, 분기 → 전체순번(선택 사항, 여유 있으면 사용)
    df["time_idx"] = (df["년도"] - df["년도"].min()) * 4 + (df["분기"] - 1)
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # 동별로 groupby하여 lag, rolling 생성
    df = df.copy()
    df = df.sort_values(["행정동", "년도", "분기"])

    # lag 1, lag 4 생성 (1분기 전, 1년 전)
    for col in TARGET_COLS:
        df[f"{col}_lag1"] = df.groupby("행정동")[col].shift(1)
        df[f"{col}_lag4"] = df.groupby("행정동")[col].shift(4)

    # 최근 4분기 평균 (길단위유동인구 기준)
    df["길단위유동인구_roll4"] = (
        df.groupby("행정동")["길단위유동인구"]
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

   
    # lag 때문에 생긴 NaN 제거 (초기 몇 분기)
    df = df.dropna().reset_index(drop=True)
    print("-------------- add_lag_features ------------------")
    print(df.head())
    return df

# -----------------------------
# 4. 범주형 인코딩
# -----------------------------
def encode_categoricals(df: pd.DataFrame):
    df = df.copy()
    le_gu = LabelEncoder()
    le_dong = LabelEncoder()

    df["행정구_le"] = le_gu.fit_transform(df["행정구"])
    df["행정동_le"] = le_dong.fit_transform(df["행정동"])

    return df, le_gu, le_dong

# -----------------------------
# 5. 학습 데이터셋 구성
# -----------------------------
def build_dataset(df: pd.DataFrame):
    """
    df: lag/rolling, 인코딩까지 끝난 상태
    """
    # feature 컬럼 정의
    feature_cols = [
        "행정구_le",
        "행정동_le",
        "년도",
        "분기",
        "time_idx",
        "길단위유동인구_roll4",
    ]

    # lag feature 추가
    for col in TARGET_COLS:
        feature_cols.append(f"{col}_lag1")
        feature_cols.append(f"{col}_lag4")

    X = df[feature_cols].copy()
    y = df[TARGET_COLS].copy()  # Multi-output (6개)

    return X, y, feature_cols

# -----------------------------
# 6. 모델 학습
# -----------------------------
def train_model(X: pd.DataFrame, y: pd.DataFrame):
    # 시간 정보 보존을 위해 단순 랜덤 split보단
    # 최근 연도를 test로 쓰는 게 좋지만, 예시는 그냥 split 사용
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 간단한 성능 확인 (R^2)
    r2_train = model.score(X_train, y_train)
    r2_val = model.score(X_val, y_val)
    print(f"[RF] Train R^2: {r2_train:.4f}")
    print(f"[RF] Valid R^2: {r2_val:.4f}")

    return model

# -----------------------------
# 7. 학습 전체 파이프라인
# -----------------------------
def train_and_save_model():
    df = load_data(DATA_PATH)
    df = add_time_features(df)
    df = add_lag_features(df)
    df, le_gu, le_dong = encode_categoricals(df)
    X, y, feature_cols = build_dataset(df)

    model = train_model(X, y)

    # 모델 + 인코더 + feature_cols 저장
    artifact = {
        "model": model,
        "le_gu": le_gu,
        "le_dong": le_dong,
        "feature_cols": feature_cols,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

    # 원본 데이터(예측용 history로 사용)도 같이 리턴
    return df, artifact

# -----------------------------
# 8. 특정 동의 미래 분기 예측 함수
# -----------------------------
def next_quarter(year: int, quarter: int):
    """다음 분기 계산 (예: 2025,4 → 2026,1)"""
    if quarter == 4:
        return year + 1, 1
    else:
        return year, quarter + 1

def predict_future_for_dong(
    df_raw: pd.DataFrame,
    artifact: dict,
    gu_name: str,
    dong_name: str,
    start_year: int,
    start_quarter: int,
    n_steps: int = 2,
):
    """
    df_raw: lag 계산 전 원본 데이터(행정구, 행정동, 년도, 분기, 타겟 포함)
    artifact: joblib에서 불러온 모델/인코더
    gu_name, dong_name: 예) "강남구", "개포1동"
    start_year, start_quarter: 예) 2026, 1
    n_steps: 예측할 분기 수 (2 → 1분기, 2분기)
    """

    model = artifact["model"]
    le_gu = artifact["le_gu"]
    le_dong = artifact["le_dong"]
    feature_cols = artifact["feature_cols"]

    # 해당 동의 과거 데이터만 추출
    hist = df_raw[(df_raw["행정구"] == gu_name) & (df_raw["행정동"] == dong_name)].copy()
    hist = hist.sort_values(["년도", "분기"]).reset_index(drop=True)

    if hist.empty:
        raise ValueError(f"{gu_name} {dong_name} 데이터가 없습니다.")

    results = []

    cur_year = start_year
    cur_quarter = start_quarter

    for step in range(n_steps):
        # --- lag/rolling 계산용으로 최근 데이터 기준 ---
        # 최소한 1개는 있다고 가정
        last_rows = hist.tail(4)  # 최근 최대 4분기

        row = {}

        # 인코딩된 구/동
        row["행정구_le"] = le_gu.transform([gu_name])[0]
        row["행정동_le"] = le_dong.transform([dong_name])[0]

        # 년도, 분기, time_idx
        row["년도"] = cur_year
        row["분기"] = cur_quarter
        row["time_idx"] = (cur_year - df_raw["년도"].min()) * 4 + (cur_quarter - 1)

        # rolling 4분기 평균 (길단위유동인구)
        row["길단위유동인구_roll4"] = last_rows["길단위유동인구"].mean()

        # lag1 = 직전 분기, lag4 = 1년 전(4분기 전)
        for col in TARGET_COLS:
            # lag1
            row[f"{col}_lag1"] = last_rows[col].iloc[-1]

            # lag4 (4개 미만이면 그냥 평균 사용)
            if len(last_rows) >= 4:
                row[f"{col}_lag4"] = last_rows[col].iloc[0]
            else:
                row[f"{col}_lag4"] = last_rows[col].mean()

        # feature vector 만들기
        X_future = pd.DataFrame([row])[feature_cols]
        y_pred = model.predict(X_future)[0]  # shape: (6,)

        pred_dict = {
            "행정구": gu_name,
            "행정동": dong_name,
            "년도": cur_year,
            "분기": cur_quarter,
        }
        for col, val in zip(TARGET_COLS, y_pred):
            pred_dict[col] = float(val)

        results.append(pred_dict)

        # 예측값을 hist에 붙여서 다음 분기 예측의 lag로 사용 (auto-regressive)
        # 실제 데이터와 같은 컬럼 구조로 맞춰줌
        new_hist_row = {
            "행정구": gu_name,
            "행정동": dong_name,
            "년도": cur_year,
            "분기": cur_quarter,
        }
        for col in TARGET_COLS:
            new_hist_row[col] = float(pred_dict[col])

        hist = pd.concat([hist, pd.DataFrame([new_hist_row])], ignore_index=True)

        # 다음 분기로 이동
        cur_year, cur_quarter = next_quarter(cur_year, cur_quarter)

    return pd.DataFrame(results)

# -----------------------------
# 9. 모델 로드 함수
# -----------------------------
def load_model(model_path: str = MODEL_PATH):
    """저장된 모델을 로드합니다."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}\n"
            f"먼저 train_and_save_model()을 실행하여 모델을 학습하고 저장하세요."
        )
    artifact = joblib.load(model_path)
    print(f"모델 로드 완료: {model_path}")
    return artifact

# -----------------------------
# 10. 실행 예시
# -----------------------------
if __name__ == "__main__":
    # 1) 저장된 모델 로드 (모델 학습 없이)
    artifact = load_model(MODEL_PATH)

    # 2) 예측 테스트: 예) 강남구 개포1동에 대해 2026년 1분기, 2분기 예측
    #    (실제 데이터의 최대 년도/분기는 엑셀을 확인해서 입력)
    df_raw = load_data(DATA_PATH)  # lag 전 원본 (history 용)

    gu = "송파구"
    dong = "가락1동"
    start_year = 2026
    start_quarter = 1
    n_steps = 3 # 1분기 + 2분기

    future_df = predict_future_for_dong(
        df_raw=df_raw,
        artifact=artifact,
        gu_name=gu,
        dong_name=dong,
        start_year=start_year,
        start_quarter=start_quarter,
        n_steps=n_steps,
    )

    print("\n=== 예측 결과 ===")
    print(future_df)

    