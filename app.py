from flask import Flask
import pandas as pd
import joblib
from pathlib import Path
from flask import request, jsonify




app = Flask(__name__)


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


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 기본 정렬: 행정동, 년도, 분기
    df = df.sort_values(["행정동", "년도", "분기"]).reset_index(drop=True)
    return df

# -----------------------------
# 8. 특정 동의 미래 분기 예측 함수
# -----------------------------
def next_quarter(year: int, quarter: int):
    """다음 분기 계산 (예: 2025,4 → 2026,1)"""
    if quarter == 4:
        return year + 1, 1
    else:
        return year, quarter + 1

 # 1) 저장된 모델 로드 (모델 학습 없이)
artifact = load_model(MODEL_PATH)
df_raw = load_data(DATA_PATH)  # lag 전 원본 (history 용)


@app.route("/predict-data", methods=["GET"])
def predict_data():

    # 쿼리 파라미터에서 값 읽기
    gu = request.args.get("gu")
    dong = request.args.get("dong")
    start_year = int(request.args.get("start_year"))
    start_quarter = int(request.args.get("start_quarter"))
    n_steps = int(request.args.get("n_steps"))

    # 2) 예측 테스트: 예) 강남구 개포1동에 대해 2026년 1분기, 2분기 예측
    #    (실제 데이터의 최대 년도/분기는 엑셀을 확인해서 입력)
   

    print(f"gu: {gu}, dong: {dong}, start_year: {start_year}, start_quarter: {start_quarter}, n_steps: {n_steps}")
    future_df = predict_future_for_dong(
        df_raw=df_raw,
        artifact=artifact,
        gu_name=gu,
        dong_name=dong,
        start_year=start_year,
        start_quarter=start_quarter,
        n_steps=n_steps,
    )
    print(f"future_df: {future_df}")
  
    # 한글이 깨지지 않도록 파이썬 객체로 변환 후 jsonify 사용
    result = future_df.to_dict(orient="records")
    print(f"result: {result}")
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)


# ##점수 code##
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # pkl 불러오기
# model = joblib.load("lizhi_PCA_score_model.pkl")

# X_cols = model["X_cols"]
# year_weights = model["year_weights"]
# pca_full = model["pca_full"]

# # 스케일러는 전체 데이터 기준으로 생성
# scaler = StandardScaler().fit(pca_full[X_cols])

# # PCA loadings 평균값 사용
# pca_weights = pca_full[[f"PCA_weight_{c}" for c in X_cols]].mean().values


# @app.route("/predict_score", methods=["POST"])
# def predict_score():
#     data = request.json
#     x = pd.DataFrame([data])

#     # 점포 경쟁도 변환
#     x["전체_역점포"] = 1/(x["전체점포수"]+1)
#     x["음식점_역점포"] = 1/(x["음식점점포수"]+1)
#     x["카페_역점포"]   = 1/(x["카페점포수"]+1)
#     x["호프_역점포"]   = 1/(x["호프점포수"]+1)

#     x = x[X_cols]

#     # 스케일링
#     x_scaled = scaler.transform(x)

#     # PCA PC1 계산
#     pc1 = np.dot(x_scaled, pca_weights)

#     # 전체 데이터 대비 percentile 변환 (0~100)
#     base_pc1 = pca_full["PC1_raw"].values
#     percentile = (pc1 < base_pc1).mean() * 100

#     return jsonify({
#         "입지점수(0~100)": round(float(percentile), 4)
#     })


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

