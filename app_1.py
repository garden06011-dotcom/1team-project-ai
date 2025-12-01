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

# ---------------------------------------------------------
# 2. /predict (업종별 개별 X값 직접 예측)
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    shop_type = data.get("type")  # cafe / hansic / hof

    if shop_type not in models:
        return jsonify({"error": "type은 cafe, hansic, hof 중 하나여야 함"})

    model = models[shop_type]

    try:
        X_input = [[
            data["정규화매출효율"],
            data["정규화성장률"],
            data["정규화경쟁점수"],
            data["작년 매출"],
            data["이전 매출"],
            data["작년 점포수"],
            data["이전 점포수"]
        ]]

        y_pred = model.predict(X_input)[0]
        return jsonify({
            "업종": shop_type,
            "예측Y": float(round(y_pred, 4))
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------------------------------------
# 3. /score?dong=OO&type=OO  (업종 1개 점수)
# ---------------------------------------------------------
@app.route("/score", methods=["GET"])
def score():
    dong = request.args.get("dong")
    shop_type = request.args.get("type")  # cafe / hansic / hof

    if shop_type not in dong_scores:
        return jsonify({"error": "type은 cafe, hansic, hof 중 하나여야 함"})

    scores = dong_scores[shop_type]

    if dong in scores.index:
        val = scores.loc[dong, "동별_평균점수"]
        return jsonify({
            "dong": dong,
            "type": shop_type,
            "score": float(round(val, 4))
        })
    else:
        return jsonify({"error": f"{dong} 동을 찾을 수 없습니다."})

# ---------------------------------------------------------
# 4. /score_all?dong=OO (업종 3개 점수 한번에!)
# ---------------------------------------------------------
@app.route("/score_all", methods=["GET"])
def score_all():
    dong = request.args.get("dong")

    result = {"dong": dong}

    for shop_type, df_scores in dong_scores.items():
        if dong in df_scores.index:
            result[shop_type] = float(round(df_scores.loc[dong, "동별_평균점수"], 4))
        else:
            result[shop_type] = None

    return jsonify(result)

# ---------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
