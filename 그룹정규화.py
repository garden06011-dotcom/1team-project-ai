"""
동-임대료 구간별 그룹 정규화 스크립트
- 임대료를 5개 구간(최저가/저가/중가/고가/최고가)으로 분류
- 동별 임대료 구간별로 매출효율, 성장률, 경쟁밀도 정규화
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 설정
# ============================================================
INPUT_FILE = "data/호프_구간정규화_최종.xlsx"
OUTPUT_FILE = "data/호프_그룹정규화_결과.xlsx"

# 임대료 구간 정의
RENT_LEVELS = {
    1: "최저가",
    2: "저가", 
    3: "중가",
    4: "고가",
    5: "최고가"
}

# 정규화할 컬럼
NORMALIZE_COLS = ['매출효율', '성장률', '경쟁밀도']

# ============================================================
# 1. 임대료 구간 생성 함수
# ============================================================
def create_rent_level(df):
    """
    동별로 임대료를 5개 구간으로 분류
    
    각 동 내에서:
    - 최저가: 하위 0~20%
    - 저가: 20~40%
    - 중가: 40~60%
    - 고가: 60~80%
    - 최고가: 80~100%
    """
    print("\n📊 임대료 구간 생성 중...")
    
    def assign_rent_level(group):
        """각 동 그룹별로 임대료 구간 할당"""
        # 임대료가 NaN이면 중간값으로 처리
        rent = group['임대료'].fillna(group['임대료'].median())
        
        # 고유값이 5개 미만이면 간단히 처리
        unique_values = rent.nunique()
        
        if unique_values < 5:
            # 고유값이 적으면 간단히 구간 할당
            if unique_values == 1:
                group['임대료_구간'] = 3  # 모두 중가
            else:
                # 값을 정렬하여 순서대로 구간 할당
                group['임대료_구간'] = pd.cut(
                    rent,
                    bins=unique_values,
                    labels=list(range(1, unique_values + 1)),
                    duplicates='drop'
                )
        else:
            try:
                # Quantile 기반 구간 분류
                group['임대료_구간'] = pd.qcut(
                    rent, 
                    q=5, 
                    labels=[1, 2, 3, 4, 5],
                    duplicates='drop'
                )
            except:
                # 실패 시 동일 너비 구간으로 분류
                group['임대료_구간'] = pd.cut(
                    rent,
                    bins=5,
                    labels=[1, 2, 3, 4, 5],
                    duplicates='drop'
                )
        
        # NaN 처리
        group['임대료_구간'] = group['임대료_구간'].fillna(3)
        
        return group
    
    # 동별로 그룹화하여 임대료 구간 생성
    df_with_level = df.groupby('행정동명', group_keys=False).apply(assign_rent_level)
    
    # 구간 레이블 추가
    df_with_level['임대료_구간_명'] = df_with_level['임대료_구간'].map(RENT_LEVELS)
    
    print(f"   ✅ 임대료 구간 생성 완료")
    print(f"   - 구간 분포:")
    print(df_with_level['임대료_구간_명'].value_counts().sort_index())
    
    return df_with_level

# ============================================================
# 2. Min-Max 정규화 함수
# ============================================================
def min_max_normalize(series):
    """Min-Max 정규화 (0~100 범위)"""
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([50] * len(series), index=series.index)
    
    normalized = 100 * (series - min_val) / (max_val - min_val)
    return normalized

# ============================================================
# 3. 그룹별 정규화 함수
# ============================================================
def group_normalize(df, group_cols, normalize_cols):
    """
    그룹별 정규화 수행
    
    Args:
        df: 데이터프레임
        group_cols: 그룹화 기준 컬럼 (예: ['행정동명', '임대료_구간'])
        normalize_cols: 정규화할 컬럼 리스트
    """
    print(f"\n🔧 그룹 정규화 시작...")
    print(f"   - 그룹 기준: {group_cols}")
    print(f"   - 정규화 컬럼: {normalize_cols}")
    
    df_normalized = df.copy()
    
    for col in normalize_cols:
        print(f"\n   📈 {col} 정규화 중...")
        
        # 경쟁밀도는 낮을수록 좋으므로 역정규화 필요
        if '경쟁밀도' in col:
            print(f"      → 경쟁밀도 역지표 적용")
            # 그룹별 정규화
            normalized = df_normalized.groupby(group_cols)[col].transform(
                lambda x: 100 - min_max_normalize(x)
            )
        else:
            # 그룹별 정규화
            normalized = df_normalized.groupby(group_cols)[col].transform(
                min_max_normalize
            )
        
        # 새 컬럼명 생성
        new_col_name = f'그룹정규화_{col}'
        df_normalized[new_col_name] = normalized
        
        print(f"      ✅ {new_col_name} 생성")
        print(f"         범위: {normalized.min():.2f} ~ {normalized.max():.2f}")
    
    return df_normalized

# ============================================================
# 4. 그룹별 통계 확인
# ============================================================
def show_group_statistics(df, group_cols, normalize_cols):
    """그룹별 정규화 통계 출력"""
    print("\n" + "=" * 70)
    print("📊 그룹별 정규화 통계")
    print("=" * 70)
    
    # 샘플 동 선택
    sample_dong = df['행정동명'].iloc[0]
    sample_data = df[df['행정동명'] == sample_dong]
    
    print(f"\n📍 샘플 동: {sample_dong}")
    print(f"   총 데이터: {len(sample_data)}행")
    
    # 구간별 통계
    for rent_level in sample_data['임대료_구간'].unique():
        level_name = RENT_LEVELS.get(rent_level, f"구간{rent_level}")
        level_data = sample_data[sample_data['임대료_구간'] == rent_level]
        
        print(f"\n   💰 {level_name} (구간 {rent_level}):")
        print(f"      데이터 수: {len(level_data)}행")
        
        for col in normalize_cols:
            original_col = col
            normalized_col = f'그룹정규화_{col}'
            
            if original_col in level_data.columns and normalized_col in level_data.columns:
                print(f"      {col}:")
                print(f"         원본: {level_data[original_col].mean():.2f} "
                      f"(범위: {level_data[original_col].min():.2f}~{level_data[original_col].max():.2f})")
                print(f"         정규화: {level_data[normalized_col].mean():.2f} "
                      f"(범위: {level_data[normalized_col].min():.2f}~{level_data[normalized_col].max():.2f})")

# ============================================================
# 5. 정규화 전후 비교
# ============================================================
def compare_normalization(df, normalize_cols):
    """기존 정규화 vs 그룹 정규화 비교"""
    print("\n" + "=" * 70)
    print("🔍 정규화 방식 비교")
    print("=" * 70)
    
    for col in normalize_cols:
        # 기존 정규화 컬럼명
        old_col = f'정규화 {col}' if '정규화' not in col else col
        new_col = f'그룹정규화_{col}'
        
        if old_col in df.columns and new_col in df.columns:
            print(f"\n📊 {col}:")
            print(f"   [기존 정규화] 평균: {df[old_col].mean():.2f}, "
                  f"표준편차: {df[old_col].std():.2f}")
            print(f"   [그룹 정규화] 평균: {df[new_col].mean():.2f}, "
                  f"표준편차: {df[new_col].std():.2f}")
            
            # 상관계수
            corr = df[[old_col, new_col]].corr().iloc[0, 1]
            print(f"   상관계수: {corr:.4f}")

# ============================================================
# 6. Y점수 재계산
# ============================================================
def recalculate_y_score(df):
    """그룹 정규화 기반 Y점수 재계산"""
    print("\n🎯 Y점수 재계산 중...")
    
    # 기존 가중치 사용 (동일)
    df['Y점수_그룹정규화'] = (
        df['그룹정규화_매출효율'] * 0.4 + 
        df['그룹정규화_성장률'] * 0.3 + 
        df['그룹정규화_경쟁밀도'] * 0.3
    )
    
    print(f"   ✅ Y점수 재계산 완료")
    print(f"      평균: {df['Y점수_그룹정규화'].mean():.2f}")
    print(f"      범위: {df['Y점수_그룹정규화'].min():.2f} ~ {df['Y점수_그룹정규화'].max():.2f}")
    
    # 기존 Y점수와 비교
    if 'Y점수' in df.columns:
        corr = df[['Y점수', 'Y점수_그룹정규화']].corr().iloc[0, 1]
        print(f"      기존 Y점수와 상관계수: {corr:.4f}")
    
    return df

# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 70)
    print("🚀 동-임대료 구간별 그룹 정규화 시작")
    print("=" * 70)
    
    # 1. 데이터 로드
    print(f"\n📂 데이터 로드: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    df['임대료'] = (
    df['임대료']
        .astype(str)
        .str.replace(",", "")
        .str.strip()
)
    df['임대료'] = pd.to_numeric(df['임대료'], errors="coerce")
    print(f"   - Shape: {df.shape}")
    print(f"   - 고유 동 수: {df['행정동명'].nunique()}")
    
    # 2. 임대료 구간 생성
    df = create_rent_level(df)
    
    # 3. 그룹별 정규화 수행
    group_cols = ['행정동명', '임대료_구간']
    df = group_normalize(df, group_cols, NORMALIZE_COLS)
    
    # 4. Y점수 재계산
    df = recalculate_y_score(df)
    
    # 5. 통계 확인
    show_group_statistics(df, group_cols, NORMALIZE_COLS)
    compare_normalization(df, NORMALIZE_COLS)
    
    # 6. 결과 저장
    print(f"\n💾 결과 저장: {OUTPUT_FILE}")
    
    # 컬럼 순서 정리
    important_cols = [
        '연도', '분기', '행정동코드', '구', '행정동명',
        '임대료', '임대료_구간', '임대료_구간_명',
        '매출', '작년 매출', '이전 매출',
        '총 점포수', '작년 점포수', '이전 점포수',
        '매출효율', '그룹정규화_매출효율',
        '성장률', '그룹정규화_성장률',
        '경쟁밀도', '그룹정규화_경쟁밀도',
        'Y점수', 'Y점수_그룹정규화'
    ]
    
    # 존재하는 컬럼만 선택
    output_cols = [col for col in important_cols if col in df.columns]
    
    # 나머지 컬럼 추가
    remaining_cols = [col for col in df.columns if col not in output_cols]
    output_cols.extend(remaining_cols)
    
    df[output_cols].to_excel(OUTPUT_FILE, index=False)
    
    print(f"   ✅ 저장 완료!")
    print(f"   - 총 컬럼 수: {len(output_cols)}")
    
    # 7. 샘플 결과 출력
    print("\n" + "=" * 70)
    print("📋 샘플 결과 (처음 5행)")
    print("=" * 70)
    
    sample_cols = [
        '행정동명', '임대료_구간_명', 
        '매출효율', '그룹정규화_매출효율',
        'Y점수', 'Y점수_그룹정규화'
    ]
    print(df[sample_cols].head())
    
    print("\n" + "=" * 70)
    print("✅ 그룹 정규화 완료!")
    print("=" * 70)
    
    return df

if __name__ == "__main__":
    df_result = main()