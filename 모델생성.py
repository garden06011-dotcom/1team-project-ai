import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import joblib

# ======================================================
# 1. 데이터 로드
# ======================================================
FILE_PATH = "data/호프_정렬+정규화(결측치 완).xlsx"

df = pd.read_excel(FILE_PATH)

# 컬럼 공백 제거
df.columns = df.columns.str.strip().str.replace("\u00A0", "", regex=False)

print("📌 데이터 로드 완료:", df.shape)

# ======================================================
# 2. Feature(X) / Target(y) 정의
# ======================================================

X_cols = [
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
    "분기"
]

y_col = "Y점수 정규화"

# ======================================================
# 3. Train / Test Split
# ======================================================

train_df = df[df["연도"] < 2024]                   # 2022~2023 전체
train_df_q = df[(df["연도"] == 2024) & (df["분기"] <= 3)]  # 2024 Q1~Q3
train_df = pd.concat([train_df, train_df_q])      # 전체 학습셋 구성

test_df  = df[(df["연도"] == 2024) & (df["분기"] == 4)]    # 검증: 2024 Q4

X_train, y_train = train_df[X_cols], train_df[y_col]
X_test,  y_test  = test_df[X_cols],  test_df[y_col]

print("📌 Train:", X_train.shape, "/ Test:", X_test.shape)

# ======================================================
# 4. LightGBM 모델 학습
# ======================================================

print("\n🚀 LightGBM 모델 학습 시작...")

lgb_model = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(X_train, y_train)

print("✅ 모델 학습 완료")

# ======================================================
# 5. 성능 평가
# ======================================================

preds = lgb_model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)

print("\n📊 ===== 모델 성능 (2024 Q4 예측) =====")
print("MAE:", mae)
print("R² :", r2)

# ======================================================
# 6. Feature Importance 출력
# ======================================================

print("\n📌 Feature Importance:")
for name, importance in sorted(zip(X_cols, lgb_model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{name:<20} : {importance}")

# ======================================================
# 7. 모델 저장
# ======================================================

MODEL_PATH = "models/lgb_2025_model_hof.pkl"
joblib.dump(lgb_model, MODEL_PATH)

print(f"\n🎉 모델 저장 완료 → {MODEL_PATH}")
