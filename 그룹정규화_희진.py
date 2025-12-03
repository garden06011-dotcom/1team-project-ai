import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib


# ==========================================================
# 1) 데이터 로드
# ==========================================================
df = pd.read_excel("카페_정렬+정규화.xlsx")

# 컬럼명 앞뒤 공백 / 특수공백 제거 (중간 공백은 유지)
df.columns = df.columns.str.strip().str.replace("\u00A0", "", regex=False)

X_cols = [
    "정규화매출효율", "정규화성장률", "정규화경쟁밀도",
    "매출", "작년 매출", "이전 매출",
    "총 점포수", "작년 점포수", "이전 점포수",
    "임대료",
    "연도", "분기"
]
y_col = "Y점수 정규화"

train_df = df[df["연도"] <= 2023]
test_df  = df[df["연도"] == 2024]

X_train, y_train = train_df[X_cols], train_df[y_col]
X_test,  y_test  = test_df[X_cols],  test_df[y_col]

print("\n=== 데이터 로드 완료 ===")
print("Train:", X_train.shape, "/ Test:", X_test.shape)


# ==========================================================
# 2) XGBoost & LightGBM 기본 모델 학습
# ==========================================================
print("\n=== 기본 모델 학습 중... ===")

xgb_best = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_best.fit(X_train, y_train)
xgb_best_pred = xgb_best.predict(X_test)

lgb_best = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_best.fit(X_train, y_train)
lgb_best_pred = lgb_best.predict(X_test)

print("\n===== XGBoost 성능 =====")
print("MAE:", mean_absolute_error(y_test, xgb_best_pred))
print("R² :", r2_score(y_test, xgb_best_pred))

print("\n===== LightGBM 성능 =====")
print("MAE:", mean_absolute_error(y_test, lgb_best_pred))
print("R² :", r2_score(y_test, lgb_best_pred))


# ==========================================================
# 3) 2025 예측 함수
# ==========================================================
def predict_2025(dong_name, model):
    # 해당 동 데이터만 정렬
    dong_df = df[df["행정동명"] == dong_name].sort_values(["연도", "분기"])

    if dong_df.empty:
        raise ValueError(f"{dong_name} 동 데이터가 없습니다.")

    last = dong_df.iloc[-1]   # 마지막 행 (보통 2024년 4분기라고 가정)

    future = pd.DataFrame([
        {
            "정규화매출효율": last["정규화매출효율"],
            "정규화성장률": last["정규화성장률"],
            "정규화경쟁밀도": last["정규화경쟁밀도"],
            "매출": last["매출"],
            "작년 매출": last["작년 매출"],
            "이전 매출": last["이전 매출"],
            "총 점포수": last["총 점포수"],
            "작년 점포수": last["작년 점포수"],
            "이전 점포수": last["이전 점포수"],
            "임대료": last["임대료"],
            "연도": 2025, "분기": 1
        },
        {
            "정규화매출효율": last["정규화매출효율"],
            "정규화성장률": last["정규화성장률"],
            "정규화경쟁밀도": last["정규화경쟁밀도"],
            "매출": last["매출"],
            "작년 매출": last["작년 매출"],
            "이전 매출": last["이전 매출"],
            "총 점포수": last["총 점포수"],
            "작년 점포수": last["작년 점포수"],
            "이전 점포수": last["이전 점포수"],
            "임대료": last["임대료"],
            "연도": 2025, "분기": 2
        }
    ])[X_cols]

    pred = model.predict(future)
    return pred


# 테스트용: 특정 동 2025년 예측
dong = "청운효자동"   # 원하는 동 이름으로 바꿔도 됨
xgb_future = predict_2025(dong, xgb_best)
lgb_future = predict_2025(dong, lgb_best)

print(f"\n===== 2025년 예측 ({dong}) =====")
print("XGBoost 2025 Q1/Q2:", xgb_future)
print("LightGBM 2025 Q1/Q2:", lgb_future)


# ==========================================================
# 4) 모델 저장
# ==========================================================
joblib.dump(xgb_best, "best_xgb_model.pkl")
joblib.dump(lgb_best, "best_lgb_model.pkl")

print("\n🎉 최적(기본) 모델 저장 완료!  (best_xgb_model.pkl / best_lgb_model.pkl)")
