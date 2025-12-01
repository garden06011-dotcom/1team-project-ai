"""
업종별 매출 데이터 필터링 및 면적 데이터 추가 스크립트
1. 남은_행정동.xlsx의 CD 기준으로 각 업종 데이터 필터링
2. 필터링된 데이터에 면적 정보 추가
3. 컬럼명을 한글로 번역
"""

import pandas as pd
import os

# 컬럼명 한글 번역 매핑
COLUMN_TRANSLATION = {
    # 기준 컬럼
    'year': '연도',
    'quarter': '분기',
    'CD': '행정동코드',
    'NM': '행정동명',
    'GUBUN': '구분',
    
    # 매출 컬럼 (업종별로 다름)
    '카페': '카페매출',
    '한식음식점': '한식매출',
    '호프집': '호프매출',
    
    # 구분자 (삭제 예정)
    '__population_separator__': None,
    '__store_separator__': None,
    '__store_hansik_separator__': None,
    '__rent_separator__': None,
    
    # 인구 데이터 (1분기)
    'TOT_FLPOP_CO_1': '1분기_총유동인구',
    'TOT_BDPOP_CO_1': '1분기_건물내유동인구',
    'TOT_REPOP_CO_1': '1분기_거주인구',
    'TOT_WRC_POPLTN_CO_1': '1분기_직장인구',
    
    # 인구 데이터 (2분기)
    'TOT_FLPOP_CO_2': '2분기_총유동인구',
    'TOT_BDPOP_CO_2': '2분기_건물내유동인구',
    'TOT_REPOP_CO_2': '2분기_거주인구',
    'TOT_WRC_POPLTN_CO_2': '2분기_직장인구',
    
    # 인구 데이터 (3분기)
    'TOT_FLPOP_CO_3': '3분기_총유동인구',
    'TOT_BDPOP_CO_3': '3분기_건물내유동인구',
    'TOT_REPOP_CO_3': '3분기_거주인구',
    'TOT_WRC_POPLTN_CO_3': '3분기_직장인구',
    
    # 점포 데이터 (브랜드1 - 스타벅스 등)
    'FIRST_TOT': '브랜드1_총점포수',
    'FIRST_NOR': '브랜드1_일반점포',
    'FIRST_FRC': '브랜드1_프랜차이즈',
    
    # 점포 데이터 (브랜드2 - 투썸플레이스 등)
    'SECOND_TOT': '브랜드2_총점포수',
    'SECOND_NOR': '브랜드2_일반점포',
    'SECOND_FRC': '브랜드2_프랜차이즈',
    
    # 점포 데이터 (브랜드3 - 이디야 등)
    'THIRD_TOT': '브랜드3_총점포수',
    'THIRD_NOR': '브랜드3_일반점포',
    'THIRD_FRC': '브랜드3_프랜차이즈',
    
    # 임대료 데이터 (브랜드1)
    'BF1_FST_FLOOR': '브랜드1_1층임대료',
    'BF1_EX_FLOOR': '브랜드1_비1층임대료',
    'BF1_TOT_FLOOR': '브랜드1_전체평균임대료',
    
    # 임대료 데이터 (브랜드2)
    'BF2_FST_FLOOR': '브랜드2_1층임대료',
    'BF2_EX_FLOOR': '브랜드2_비1층임대료',
    'BF2_TOT_FLOOR': '브랜드2_전체평균임대료',
    
    # 임대료 데이터 (브랜드3)
    'BF3_FST_FLOOR': '브랜드3_1층임대료',
    'BF3_EX_FLOOR': '브랜드3_비1층임대료',
    'BF3_TOT_FLOOR': '브랜드3_전체평균임대료',
    
    # 면적
    '면적': '면적_km2'
}


def translate_columns(df):
    """
    데이터프레임의 컬럼명을 한글로 번역
    구분자 컬럼(__로 시작하는)은 삭제
    """
    # 구분자 컬럼 삭제
    separator_cols = [col for col in df.columns if col.startswith('__')]
    df = df.drop(columns=separator_cols)
    
    # 컬럼명 변경
    df = df.rename(columns=COLUMN_TRANSLATION)
    
    return df

def filter_and_add_area():
    """
    남은 행정동 기준으로 필터링하고 면적 데이터를 추가
    """
    # 스크립트 실행 디렉토리 (test2.py 위치)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # data 폴더 경로
    data_dir = os.path.join(script_dir, "data")
    # output 폴더 경로
    output_dir = os.path.join(script_dir, "output")
    
    # output 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 70)
    print("업종별 데이터 필터링 및 면적 추가 프로그램")
    print("=" * 70)
    print(f"\n작업 디렉토리: {script_dir}")
    print(f"데이터 폴더: {data_dir}")
    print(f"출력 폴더: {output_dir}")
    print()
    
    # 파일 경로 설정
    files = {
        '남은_행정동': os.path.join(data_dir, '남은_행정동.xlsx'),
        '면적': os.path.join(data_dir, '면적_파일.xlsx'),
        '카페': os.path.join(data_dir, '카페_매출추가.xlsx'),
        '한식': os.path.join(data_dir, '한식_매출추가.xlsx'),
        '호프': os.path.join(data_dir, '호프_매출추가.xlsx')
    }
    
    # 파일 존재 확인
    print("[파일 존재 확인]")
    for name, path in files.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {os.path.basename(path)}")
        else:
            print(f"  ✗ {name}: 파일을 찾을 수 없습니다 - {path}")
            return
    
    print("\n" + "=" * 70)
    print("1단계: 남은 행정동 코드 로드")
    print("=" * 70)
    
    # 1. 남은 행정동 파일 읽기
    df_filter = pd.read_excel(files['남은_행정동'])
    filter_codes = set(df_filter['CD'].astype(str).unique())
    print(f"✓ 남은 행정동 개수: {len(filter_codes)}개")
    print(f"  샘플 코드: {list(filter_codes)[:5]}")
    
    print("\n" + "=" * 70)
    print("2단계: 면적 데이터 전처리")
    print("=" * 70)
    
    # 2. 면적 파일 읽기 및 전처리
    df_area_raw = pd.read_excel(files['면적'])
    
    # 헤더 행 제거하고 실제 데이터만 추출 (4번째 행부터)
    df_area = df_area_raw.iloc[3:].copy()
    
    # 행정동코드가 있는 행만 선택
    df_area = df_area[df_area['행정동코드'].notna()].copy()
    
    # 필요한 컬럼만 선택하고 이름 변경
    df_area = df_area[['동별(3)', '행정동코드', '2024']].copy()
    df_area.columns = ['면적_동명', 'CD', '면적']
    
    # CD를 문자열로 변환
    df_area['CD'] = df_area['CD'].astype(float).astype(int).astype(str)
    
    # 면적을 숫자로 변환 (문자열인 경우 처리)
    df_area['면적'] = pd.to_numeric(df_area['면적'], errors='coerce')
    
    print(f"✓ 면적 데이터 전처리 완료: {len(df_area)}개 행정동")
    print(f"\n샘플 데이터:")
    print(df_area.head())
    
    print("\n" + "=" * 70)
    print("3단계: 업종별 데이터 필터링 및 면적 추가")
    print("=" * 70)
    
    # 3. 각 업종별로 필터링 및 면적 추가
    industries = {
        '카페': files['카페'],
        '한식': files['한식'],
        '호프': files['호프']
    }
    
    results = {}
    
    for industry, filepath in industries.items():
        print(f"\n[{industry}]")
        
        # 데이터 로드
        df = pd.read_excel(filepath)
        original_count = len(df)
        print(f"  원본 데이터: {original_count:,}행")
        
        # CD를 문자열로 변환
        df['CD'] = df['CD'].astype(str)
        
        # 필터링
        df_filtered = df[df['CD'].isin(filter_codes)].copy()
        filtered_count = len(df_filtered)
        removed_count = original_count - filtered_count
        
        print(f"  필터링 후: {filtered_count:,}행")
        print(f"  제거: {removed_count:,}행")
        
        # 면적 데이터 추가 (CD 기준으로 병합)
        df_with_area = df_filtered.merge(
            df_area[['CD', '면적']],
            on='CD',
            how='left'
        )
        
        # 면적이 추가된 행 개수 확인
        area_added = df_with_area['면적'].notna().sum()
        area_missing = df_with_area['면적'].isna().sum()
        
        print(f"  면적 추가: {area_added:,}행")
        if area_missing > 0:
            print(f"  ⚠️ 면적 누락: {area_missing:,}행")
        
        # 컬럼명을 한글로 번역
        df_translated = translate_columns(df_with_area)
        print(f"  컬럼명 한글 번역 완료: {len(df_translated.columns)}개 컬럼")
        
        # 결과 저장
        results[industry] = df_translated
        
        # 파일 저장
        output_path = os.path.join(output_dir, f'{industry}_최종.xlsx')
        df_translated.to_excel(output_path, index=False)
        print(f"  ✓ 저장: {os.path.basename(output_path)}")
    
    print("\n" + "=" * 70)
    print("4단계: 결과 요약")
    print("=" * 70)
    
    for industry, df in results.items():
        print(f"\n[{industry}]")
        print(f"  - 총 행 수: {len(df):,}")
        print(f"  - 총 컬럼 수: {len(df.columns)}")
        
        # 고유 행정동
        if '행정동코드' in df.columns:
            print(f"  - 고유 행정동: {df['행정동코드'].nunique()}개")
        
        # 기간
        if '연도' in df.columns and '분기' in df.columns:
            print(f"  - 기간: {df['연도'].min()}년 {df['분기'].min()}분기 ~ {df['연도'].max()}년 {df['분기'].max()}분기")
        
        # 매출 컬럼 확인
        sales_cols = [col for col in df.columns if col in ['카페매출', '한식매출', '호프매출']]
        if sales_cols:
            sales_col = sales_cols[0]
            print(f"  - 매출 데이터: {df[sales_col].notna().sum():,}개 행")
        
        # 면적 데이터
        if '면적_km2' in df.columns:
            print(f"  - 면적 데이터: {df['면적_km2'].notna().sum():,}개 행")
        
        # 컬럼 목록
        print(f"  - 컬럼: {', '.join(df.columns[:10].tolist())}...")
    
    print("\n" + "=" * 70)
    print("모든 작업이 완료되었습니다!")
    print("=" * 70)
    print(f"\n출력 파일 위치: {output_dir}")
    print("생성된 파일:")
    for industry in industries.keys():
        print(f"  - {industry}_최종.xlsx")
    
    # 샘플 데이터 출력
    print("\n" + "=" * 70)
    print("샘플 데이터 미리보기 (카페)")
    print("=" * 70)
    sample_cols = ['연도', '분기', '행정동코드', '행정동명', '카페매출', '면적_km2']
    if all(col in results['카페'].columns for col in sample_cols):
        print(results['카페'][sample_cols].head(10))
    else:
        print(results['카페'].head())


if __name__ == "__main__":
    try:
        filter_and_add_area()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n\n엔터를 눌러 종료...")