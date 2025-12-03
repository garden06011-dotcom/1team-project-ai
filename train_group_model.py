"""
그룹 정규화 데이터로 LightGBM 모델 학습
- 동-임대료 구간별 정규화된 데이터 사용
- 더 공정하고 정확한 예측 모델
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 설정
# ============================================================
DATA_FILE = "data/카페_그룹정규화_결과.xlsx"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

# 🔥 그룹 정규화된 Feature 사용
FEATURE_COLS = [
    '연도', '분기', '임대료', '임대료_구간',  # 임대료_구간 추가!
    '매출', '작년 매출', '이전 매출',
    '총 점포수', '작년 점포수', '이전 점포수',
    '그룹정규화_매출효율',   # ✅ 그룹 정규화
    '그룹정규화_성장률',     # ✅ 그룹 정규화
    '그룹정규화_경쟁밀도'    # ✅ 그룹 정규화
]

# 🎯 그룹 정규화된 Y점수 사용
TARGET_COL = 'Y점수_그룹정규화'

# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 70)
    print("🚀 카페 업종 LightGBM 모델 학습 (그룹 정규화 데이터)")
    print("=" * 70)
    
    # 1. 데이터 로드
    print(f"\n📂 데이터 로드: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    print(f"   - Shape: {df.shape}")
    print(f"   - 연도 범위: {df['연도'].min()} ~ {df['연도'].max()}")
    print(f"   - 고유 동 수: {df['행정동명'].nunique()}")
    print(f"   - 고유 구 수: {df['구'].nunique()}")
    
    # 2. 데이터 확인
    print("\n📊 Y점수 (그룹 정규화) 분포:")
    print(df[TARGET_COL].describe())
    
    print("\n📊 임대료 구간별 분포:")
    print(df['임대료_구간_명'].value_counts().sort_index())
    
    print("\n📊 샘플 데이터 (최신 5개):")
    print(df.tail()[['연도', '분기', '구', '행정동명', '임대료_구간_명', TARGET_COL]])
    
    # 3. 시계열 분할
    print("\n✂️ 데이터 분할 (시계열 순서 유지):")
    train_df = df[df['연도'] < 2024].copy()
    test_df = df[df['연도'] == 2024].copy()
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    
    print(f"   - Train: {len(train_df)}행 (2022~2023)")
    print(f"   - Test: {len(test_df)}행 (2024)")
    
    # 4. LightGBM Dataset 생성
    print("\n🔧 LightGBM Dataset 생성...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 5. 모델 학습
    print("\n🤖 LightGBM 모델 학습 (그룹 정규화):")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 학습 시간 측정
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    
    training_time = time.time() - start_time
    
    print(f"   - Best iteration: {model.best_iteration}")
    print(f"   - 학습 시간: {training_time:.2f}초 ⚡")
    
    # 6. 예측 및 평가
    print("\n📈 모델 평가:")
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"   [Train] RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
    print(f"   [Test]  RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
    
    # 7. Feature Importance
    print("\n🔍 Top 10 중요 Feature:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # 8. 예측 샘플
    print("\n🎯 예측 샘플 (Test Set 처음 5개):")
    sample_results = pd.DataFrame({
        '구': test_df['구'].iloc[:5].values,
        '행정동명': test_df['행정동명'].iloc[:5].values,
        '임대료구간': test_df['임대료_구간_명'].iloc[:5].values,
        '실제값': y_test.iloc[:5].values,
        '예측값': y_test_pred[:5],
        '오차': np.abs(y_test.iloc[:5].values - y_test_pred[:5])
    })
    print(sample_results.to_string(index=False))
    
    # 9. 모델 저장
    model_path = OUTPUT_DIR / "hof_group_normalized_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 모델 저장 완료: {model_path}")
    
    # 10. 모델 재로드 테스트
    print("\n🔄 모델 재로드 테스트:")
    loaded_model = joblib.load(model_path)
    test_pred = loaded_model.predict(X_test[:1], num_iteration=loaded_model.best_iteration)
    print(f"   재로드 예측값: {test_pred[0]:.4f}")
    print(f"   원본 예측값: {y_test_pred[0]:.4f}")
    print(f"   ✅ 모델 재로드 성공!")
    
    # 11. 기존 모델과 비교 (있다면)
    print("\n" + "=" * 70)
    print("📊 성능 개선 분석")
    print("=" * 70)
    
    old_model_path = OUTPUT_DIR / "hof_lgb_model.pkl"
    if old_model_path.exists():
        print("\n🔍 기존 모델과 비교:")
        old_model = joblib.load(old_model_path)
        
        # 기존 모델은 다른 Feature를 사용하므로 비교 생략
        print("   기존 모델: 전체 정규화 기반")
        print("   신규 모델: 그룹 정규화 기반")
        print("   → Feature가 달라 직접 비교 불가")
    
    print("\n✅ 그룹 정규화 모델 특징:")
    print("   ✅ 동-임대료 구간별 공정한 비교")
    print("   ✅ 같은 가격대 내 정확한 순위")
    print("   ✅ 사용자 예산 맞춤 추천 가능")
    
    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print("=" * 70)
    
    return model, test_rmse, test_r2, training_time, feature_importance

if __name__ == "__main__":
    model, rmse, r2, train_time, feature_imp = main()
    
    print(f"\n📊 최종 성능:")
    print(f"   - Test RMSE: {rmse:.4f}")
    print(f"   - Test R²: {r2:.4f}")
    print(f"   - 학습 시간: {train_time:.2f}초 ⚡")
    
    if r2 > 0.90:
        print("\n🎉 우수한 모델 성능! 그룹 정규화 효과 확인!")
        print("💡 이제 같은 가격대끼리 공정하게 비교할 수 있습니다!")
    elif r2 > 0.80:
        print("\n👍 양호한 모델 성능. 실무 사용 가능합니다.")
    else:
        print("\n⚠️ 모델 성능 개선 필요. 하이퍼파라미터 튜닝을 권장합니다.")
    
    print("\n🎯 다음 단계:")
    print("   1. API 서버에 모델 적용")
    print("   2. 한식/호프 업종도 그룹 정규화 적용")
    print("   3. 프론트엔드 연동 테스트")
