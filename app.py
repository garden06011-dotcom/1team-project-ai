# #################################
# 2025 예측모델 API
# #################################

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

# ============================================================
# 모델 파일 경로 설정
# ============================================================
MODEL_DIR = "models"
PREDICTION_MODEL_PATH = os.path.join(MODEL_DIR, "population_rf_model.pkl")
PREDICTION_DATA_PATH = "인구2.xlsx"

MODEL_FILES = {
    "store": "total_data.pkl"
}

TARGET_COLS = [
    "길단위유동인구",
    "주거인구",
    "전체점포수",
    "음식점점포수",
    "카페점포수",
    "호프점포수",
]

# ============================================================
# 유틸리티 함수들
# ============================================================
def load_model_data(model_type):
    """모델 파일을 로드하고 데이터를 반환"""
    if model_type not in MODEL_FILES:
        return None
    
    file_path = os.path.join(MODEL_DIR, MODEL_FILES[model_type])
    if not os.path.exists(file_path):
        return None
    
    data = joblib.load(file_path)
    return data

def df_to_dict_list(df):
    """DataFrame을 딕셔너리 리스트로 변환 (numpy 타입 처리)"""
    if df is None or df.empty:
        return []
    
    results = []
    for idx, row in df.iterrows():
        result = {}
        for col in df.columns:
            value = row[col]
            # numpy 타입을 Python 기본 타입으로 변환
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif pd.isna(value):
                value = None
            result[col] = value
        results.append(result)
    
    return results

def load_prediction_model(model_path: str = PREDICTION_MODEL_PATH):
    """예측 모델을 로드합니다."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    artifact = joblib.load(model_path)
    return artifact

def load_prediction_data(path: str) -> pd.DataFrame:
    """예측용 데이터를 로드합니다."""
    df = pd.read_excel(path)
    df = df.sort_values(["행정동", "년도", "분기"]).reset_index(drop=True)
    return df

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
    특정 동의 미래 분기를 예측합니다.
    
    Args:
        df_raw: lag 계산 전 원본 데이터
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
        last_rows = hist.tail(4)

        row = {}
        row["행정구_le"] = le_gu.transform([gu_name])[0]
        row["행정동_le"] = le_dong.transform([dong_name])[0]
        row["년도"] = cur_year
        row["분기"] = cur_quarter
        row["time_idx"] = (cur_year - df_raw["년도"].min()) * 4 + (cur_quarter - 1)
        row["길단위유동인구_roll4"] = last_rows["길단위유동인구"].mean()

        for col in TARGET_COLS:
            row[f"{col}_lag1"] = last_rows[col].iloc[-1]
            if len(last_rows) >= 4:
                row[f"{col}_lag4"] = last_rows[col].iloc[0]
            else:
                row[f"{col}_lag4"] = last_rows[col].mean()

        X_future = pd.DataFrame([row])[feature_cols]
        y_pred = model.predict(X_future)[0]

        pred_dict = {
            "행정구": gu_name,
            "행정동": dong_name,
            "년도": cur_year,
            "분기": cur_quarter,
        }
        for col, val in zip(TARGET_COLS, y_pred):
            pred_dict[col] = float(val)

        results.append(pred_dict)

        new_hist_row = {
            "행정구": gu_name,
            "행정동": dong_name,
            "년도": cur_year,
            "분기": cur_quarter,
        }
        for col in TARGET_COLS:
            new_hist_row[col] = float(pred_dict[col])

        hist = pd.concat([hist, pd.DataFrame([new_hist_row])], ignore_index=True)
        cur_year, cur_quarter = next_quarter(cur_year, cur_quarter)

    return pd.DataFrame(results)

# ============================================================
# 애플리케이션 시작 시 예측 모델 로드
# ============================================================
try:
    prediction_artifact = load_prediction_model(PREDICTION_MODEL_PATH)
    prediction_df_raw = load_prediction_data(PREDICTION_DATA_PATH)
    print("예측 모델 로드 완료")
except Exception as e:
    prediction_artifact = None
    prediction_df_raw = None
    print(f"예측 모델 로드 실패: {e}")

# ============================================================
# API 엔드포인트 - Store 데이터
# ============================================================
@app.route('/api/store', methods=['GET'])
def get_store_data():
    """
    점포수 데이터 조회
    
    Query Parameters:
        gu: 구 이름 (선택, 없으면 전체)
        type: percent(증감률) 또는 value(증감값) (기본값: percent)
        limit: 반환 개수 (기본값: 10)
    """
    try:
        data = load_model_data('store')
        if data is None:
            return jsonify({"error": "store 데이터를 찾을 수 없습니다."}), 404
        
        gu = request.args.get('gu', None)
        ranking_type = request.args.get('type', 'percent')  # percent or value
        limit = request.args.get('limit', default=10, type=int)
        
        if gu:
            # 특정 구의 Top10
            key = f"top10_{ranking_type}_by_gu"
            if key not in data or gu not in data[key]:
                return jsonify({"error": f"{gu}의 데이터를 찾을 수 없습니다."}), 404
            
            df = data[key][gu].head(limit)
            results = df_to_dict_list(df)
            
            return jsonify({
                "gu": gu,
                "ranking_type": ranking_type,
                "data": results
            }), 200
        else:
            # 전체 데이터 (증감률 기준 정렬)
            full_df = data['full']
            
            if ranking_type == 'percent':
                sorted_df = full_df.sort_values('증감률', ascending=False).head(limit)
            else:
                sorted_df = full_df.sort_values('증감값', ascending=False).head(limit)
            
            results = df_to_dict_list(sorted_df)
            
            return jsonify({
                "ranking_type": ranking_type,
                "data": results
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

# ============================================================
# API 엔드포인트 - 구 목록
# ============================================================
@app.route('/api/districts', methods=['GET'])
def get_districts():
    """
    구 목록 조회
    """
    try:
        data = load_model_data('store')
        if data is None:
            return jsonify({"error": "store 데이터를 찾을 수 없습니다."}), 404
        
        # 구 목록 추출
        gu_list = list(data['top10_percent_by_gu'].keys())
        
        return jsonify({
            "districts": sorted(gu_list)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

# ============================================================
# API 엔드포인트 - 미래 예측
# ============================================================
@app.route('/api/predict', methods=['GET'])
def predict_data():
    """
    특정 동의 미래 분기 예측
    
    Query Parameters:
        gu: 구 이름 (예: 강남구)
        dong: 동 이름 (예: 개포1동)
        start_year: 예측 시작 년도 (예: 2026)
        start_quarter: 예측 시작 분기 (예: 1)
        n_steps: 예측할 분기 수 (기본값: 2)
    """
    if prediction_artifact is None or prediction_df_raw is None:
        return jsonify({"error": "예측 모델이 로드되지 않았습니다."}), 500
    
    try:
        gu = request.args.get("gu")
        dong = request.args.get("dong")
        start_year = request.args.get("start_year", type=int)
        start_quarter = request.args.get("start_quarter", type=int)
        n_steps = request.args.get("n_steps", default=2, type=int)
        
        if not all([gu, dong, start_year, start_quarter]):
            return jsonify({
                "error": "필수 파라미터가 누락되었습니다.",
                "required": ["gu", "dong", "start_year", "start_quarter"]
            }), 400
        
        future_df = predict_future_for_dong(
            df_raw=prediction_df_raw,
            artifact=prediction_artifact,
            gu_name=gu,
            dong_name=dong,
            start_year=start_year,
            start_quarter=start_quarter,
            n_steps=n_steps,
        )
        
        result = future_df.to_dict(orient="records")
        return jsonify({
            "gu": gu,
            "dong": dong,
            "predictions": result
        }), 200
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": f"예측 중 오류 발생: {str(e)}"}), 500

# ============================================================
# 기본 엔드포인트
# ============================================================
@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    status = {
        "status": "healthy",
        "prediction_model_loaded": prediction_artifact is not None
    }
    return jsonify(status), 200

@app.route('/', methods=['GET'])
def home():
    """API 사용 가이드"""
    return jsonify({
        "message": "상권 분석 API",
        "endpoints": {
            "점포수_조회": "/api/store?gu=강남구&type=percent&limit=10",
            "구목록_조회": "/api/districts",
            "미래_예측": "/api/predict?gu=강남구&dong=개포1동&start_year=2026&start_quarter=1&n_steps=2",
            "헬스체크": "/health"
        }
    }), 200



# -----------------------------------------------------
# 업종별 모델 로드
# -----------------------------------------------------
models = {
    "cafe": joblib.load("models/lgb_2025_model_cafe.pkl"),
    "hansic": joblib.load("models/lgb_2025_model_hansic.pkl"),
    "hof": joblib.load("models/lgb_2025_model_hof.pkl"),
}

# -----------------------------------------------------
# 업종별 원본 데이터 로드 (2022~2024 실제 + 2024 Q4 예측까지 포함된 엑셀)
# -----------------------------------------------------
dataframes = {
    "cafe": pd.read_excel("data/카페_정렬+정규화(결측지 완).xlsx"),
    "hansic": pd.read_excel("data/한식_정렬+정규화(결측치 완).xlsx"),
    "hof": pd.read_excel("data/호프_정렬+정규화(결측치 완).xlsx"),
}

# -----------------------------------------------------
# Feature 목록 (모델 학습에 사용한 값)
# -----------------------------------------------------
FEATURE_COLS = [
    "정규화매출효율",
    "정규화성장률",
    "정규화경쟁밀도",
    "매출",
    "작년 매출",
    "이전 매출",
    "총 점포수",
    "작년 점포수",
    "이전 점포수",
    "임대료",
    "연도",
    "분기",
]

# -----------------------------------------------------
# 사용자 임대료 계산 함수
#  환산임대료 = (보증금 × 12%) / 12 + 월세
#  평당_임대료 = 환산임대료 / (면적㎡ / 3.3)
#  사용자_총액 = 평당_임대료 × 사용자_평수
# -----------------------------------------------------
def calculate_user_rent(deposit, monthly, area_pyeong):
    # 보증금의 월 환산
    deposit_month = (deposit * 0.12) / 12
    converted_monthly = deposit_month + monthly

    # 1평 = 3.3㎡
    area_m2 = area_pyeong * 3.3
    rent_per_pyeong = converted_monthly / (area_m2 / 3.3)

    total_rent = rent_per_pyeong * area_pyeong
    return total_rent


# -----------------------------------------------------
# /predict_custom API
#   - 쿼리: gu, type(cafe/hansic/hof), deposit, monthly, area, top_n(optional)
#   - 로직:
#       1) 사용자 임대료 계산 후, 강제 임대료_구간 산정 (1~5)
#       2) 해당 구 + 임대료_구간 데이터가 없으면 → 가장 가까운 구간으로 자동 보정
#       3) 그 구간에 포함된 모든 행정동에 대해 2025Q1 점수 예측
#       4) 점수 기준 내림차순 정렬 후 순위 부여
#       5) 상위 top_n 반환 + 1위 동의 상세 데이터 포함
# -----------------------------------------------------
@app.route("/predict_custom", methods=["GET"])
def predict_custom():
    try:
        print("=" * 50)
        print("predict_custom API 호출됨")
        print("=" * 50)
        
        # ============================
        # 1️⃣ 요청 파라미터 받기
        # ============================
        shop_type = request.args.get("type", "cafe")
        gu = request.args.get("gu")
        
        print(f"요청 파라미터:")
        print(f"  - type: {shop_type}")
        print(f"  - gu: {gu}")
        print(f"  - deposit: {request.args.get('deposit')}")
        print(f"  - monthly: {request.args.get('monthly')}")
        print(f"  - area: {request.args.get('area')}")

        deposit = request.args.get("deposit")
        monthly = request.args.get("monthly")
        area = request.args.get("area")

        # 필수 파라미터 검증
        if not gu:
            return jsonify({"error": "gu 파라미터가 필요합니다."}), 400
        if not deposit or not monthly or not area:
            return jsonify({"error": "deposit, monthly, area 파라미터가 모두 필요합니다."}), 400

        try:
            deposit = float(deposit)
            monthly = float(monthly)
            area = float(area)
        except (ValueError, TypeError):
            return jsonify({"error": "deposit, monthly, area는 숫자여야 합니다."}), 400

        if shop_type not in models:
            return jsonify({"error": "type은 cafe / hansic / hof 중 하나여야 합니다."}), 400

        df = dataframes[shop_type]
        model = models[shop_type]

        # ============================
        # 2️⃣ 사용자 임대료 재계산 (핵심!!)
        # ============================
        user_rent = calculate_user_rent(deposit, monthly, area)

        # ============================
        # 3️⃣ 구 필터링
        # ============================
        # 디버깅: 사용 가능한 구 목록 확인
        available_gu = df["구"].unique().tolist()
        print(f"요청된 구: {gu}")
        print(f"사용 가능한 구 목록: {available_gu[:10]}...")  # 처음 10개만 출력
        
        gu_df = df[df["구"] == gu].copy()

        if gu_df.empty:
            # 유사한 구 이름 찾기
            similar_gu = [g for g in available_gu if gu in str(g) or str(g) in gu]
            error_msg = f"{gu} 데이터가 없습니다."
            if similar_gu:
                error_msg += f" 유사한 구: {similar_gu[:5]}"
            return jsonify({"error": error_msg, "available_gu": available_gu[:20]}), 404

        # ============================
        # 4️⃣ 각 행정동 최신 데이터 추출
        # ============================
        latest_df = (
            gu_df
            .sort_values(["행정동코드", "연도", "분기"])
            .groupby("행정동코드")
            .tail(1)
            .reset_index(drop=True)
        )

        # ============================
        # 5️⃣ 사용자 임대료 반영하여 2025 feature 생성
        # ============================
        rows = []

        for _, r in latest_df.iterrows():
            row = {
                "정규화매출효율": r["정규화매출효율"],
                "정규화성장률": r["정규화성장률"],
                "정규화경쟁밀도": r["정규화경쟁밀도"],

                "매출": r["매출"],
                "작년 매출": r["작년 매출"],
                "이전 매출": r["이전 매출"],

                "총 점포수": r["총 점포수"],
                "작년 점포수": r["작년 점포수"],
                "이전 점포수": r["이전 점포수"],

                # ✅ 반드시 사용자 임대료로 덮어씌움
                "임대료": user_rent,

                "연도": 2025,
                "분기": 1,

                "행정동명": r["행정동명"]
            }

            rows.append(row)

        X_all = pd.DataFrame(rows)
        X_model = X_all[FEATURE_COLS]

        # ============================
        # 6️⃣ 전체 동 점수 예측
        # ============================
        preds = model.predict(X_model)
        X_all["예측_Y"] = preds

        # ============================
        # 7️⃣ 점수 순위 정렬
        # ============================
        X_all = X_all.sort_values("예측_Y", ascending=False).reset_index(drop=True)
        X_all["rank"] = X_all.index + 1

        # ============================
        # 8️⃣ 1위 동 상세 데이터 추출
        # ============================
        best = X_all.iloc[0]

        best_detail = {
            "rank": 1,
            "dong": best["행정동명"],
            "predicted_Y": float(best["예측_Y"]),
            "매출": float(best["매출"]),
            "총점포수": int(best["총 점포수"]),
            "정규화매출효율": float(best["정규화매출효율"]),
            "정규화성장률": float(best["정규화성장률"]),
            "정규화경쟁밀도": float(best["정규화경쟁밀도"])
        }

        # ============================
        # 9️⃣ Top 10 랭킹 반환
        # ============================
        ranking = X_all.head(10)[["rank", "행정동명", "예측_Y"]].to_dict(orient="records")

        result = {
            "gu": gu,
            "type": shop_type,
            "user_input": {
                "deposit": deposit,
                "monthly": monthly,
                "area": area,
                "calculated_rent": user_rent
            },
            "top10": ranking,
            "best_dong_detail": best_detail,
            "message": "✅ 사용자 임대료 반영 2025년 동별 Y점수 예측 완료"
        }
        
        print(f"predict_custom API 성공: {len(ranking)}개 동 반환")
        print("=" * 50)
        
        return jsonify(result)

    except Exception as e:
        print(f"predict_custom API 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        return jsonify({"error": f"예측 중 오류 발생: {str(e)}"}), 500





# ---------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
