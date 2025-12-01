from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------------------------------------------------
# 1. 업종별 모델 & 점수 파일 로드
# ---------------------------------------------------------
models = {
    "cafe": joblib.load("models/cafe_XGBoost.pkl"),
    "korean": joblib.load("models/hansic_XGBoost.pkl"),
    "hof": joblib.load("models/hof_XGBoost.pkl")
}

dong_scores = {
    "cafe": pd.read_excel("data/cafe_XGBoost.xlsx", index_col=0),
    "korean": pd.read_excel("data/hansic_XGBoost.xlsx", index_col=0),
    "hof": pd.read_excel("data/hof_XGBoost.xlsx", index_col=0)
}

# 🔥 업종별 원본 데이터 (X값 반환하기 위해 반드시 필요)
original_data = {
    "cafe": pd.read_excel("data/y추가완료_카페 데이터칼럼.xlsx"),
    "korean": pd.read_excel("data/y추가완료_한식 데이터칼럼.xlsx"),
    "hof": pd.read_excel("data/y추가완료_호프 데이터칼럼.xlsx")
}

# ---------------------------------------------------------
# 2. /score?dong=OO&type=OO  (업종 1개 점수 + X값)
# ---------------------------------------------------------
@app.route("/score", methods=["GET"])
def score():
    dong = request.args.get("dong")
    shop_type = request.args.get("type")  # cafe / korean / hof

    if shop_type not in dong_scores:
        return jsonify({"error": "type은 cafe, korean, hof 중 하나여야 함"})

    # --- 1) 업종별 점수 데이터 ---
    scores = dong_scores[shop_type]

    if dong not in scores.index:
        return jsonify({"error": f"{dong} 동을 찾을 수 없습니다."})

    score_val = float(round(scores.loc[dong, "동별_평균점수"], 4))

    # --- 2) 업종별 원본 데이터에서 X값 뽑기 ---
    df_origin = original_data[shop_type]

    # 해당 동이 여러 행이면 최신(연도+분기 가장 큰 값) 선택
    dong_rows = df_origin[df_origin["행정동명"] == dong]

    if dong_rows.empty:
        return jsonify({"error": f"{dong} 동의 X값을 찾을 수 없습니다."})

    # 최신 데이터 1개 선택
    dong_latest = dong_rows.sort_values(["연도", "분기"]).iloc[-1]

    X_values = {
        "정규화매출효율": float(dong_latest["정규화매출효율"]),
        "정규화성장률": float(dong_latest["정규화성장률"]),
        "정규화경쟁점수": float(dong_latest["정규화경쟁점수"]),
        "작년 매출": float(dong_latest["작년 매출"]),
        "이전 매출": float(dong_latest["이전 매출"]),
        "작년 점포수": int(dong_latest["작년 점포수"]),
        "이전 점포수": int(dong_latest["이전 점포수"])
    }

    # --- 3) 최종 응답 ---
    return jsonify({
        "dong": dong,
        "type": shop_type,
        "score": score_val,
        "X값": X_values
    })

# ---------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
