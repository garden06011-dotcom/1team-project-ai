from flask import Flask, request, jsonify
import joblib
from pyproj import Transformer
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# 순위 분석 데이터
human_data = joblib.load("models/origin/human_data.pkl")  # 인구 데이터
sales_data = joblib.load("models/origin/sales_data.pkl")  # 매출 데이터 
store_data = joblib.load("models/origin/store_data.pkl")  # 상점 데이터
cafe_data = joblib.load("models/origin/cafe_data.pkl")  # 카페 데이터
food_data = joblib.load("models/origin/food_data.pkl")  # 음식점 데이터
hof_data = joblib.load("models/origin/pub_data.pkl")  # 호프 데이터

# 예측 모델
human_predict = joblib.load("models/predict/human_predict.pkl")  # 인구 예측 모델
sales_predict = joblib.load("models/predict/sales_predict.pkl")  # 매출 예측 모델
store_predict = joblib.load("models/predict/store_predict.pkl")  # 상점 예측 모델
cafe_predict = joblib.load("models/predict/cafe_predict.pkl")  # 카페 예측 모델
food_predict = joblib.load("models/predict/food_predict.pkl")  # 음식점 예측 모델
hof_predict = joblib.load("models/predict/hof_predict.pkl")  # 호프 예측 모델



print(human_data)
print(sales_data)
print(store_data)
print(cafe_data)
print(food_data)
print(hof_data)

print(human_predict)
print(sales_predict)
print(store_predict)
print(cafe_predict)
print(food_predict)
print(hof_predict)



# if restaurant_predict_model is not None:
#     print("restaurant_predict_model is loaded")
# else:
#     print("restaurant_predict_model is not loaded")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
