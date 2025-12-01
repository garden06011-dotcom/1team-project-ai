"""
카페 병합 파일 필터링 스크립트
남은 행정동에 있는 CD만 남기고 나머지 삭제
"""

import pandas as pd
import os

def filter_cafe_data():
    """
    카페 병합 파일에서 남은 행정동에 있는 CD만 남기고 나머지 삭제
    """
    # 스크립트 파일의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트 디렉토리 (golmokSeoul_finish의 상위 디렉토리)
    project_root = os.path.dirname(script_dir)
    
    print("=" * 70)
    print("카페 데이터 필터링 프로그램")
    print("=" * 70)
    print()
    
    # 기본 경로 설정
    default_filter_file = os.path.join(script_dir, "data", "남은_행정동2.xlsx")
    default_cafe_file = os.path.join(script_dir, "data", "한식 결합문서.xlsx")
    default_output_file = os.path.join(script_dir, "data", "한식_결합문서_필터링.xlsx")
    
    # 1. 남은_행정동2.xlsx 파일 경로 입력
    print("[1단계] 남은_행정동2.xlsx 파일 경로를 입력하세요")
    print(f"기본값: {default_filter_file}")
    print("엔터를 누르면 기본값을 사용합니다.")
    user_input = input("경로 입력: ").strip().replace('"', '')
    
    if not user_input:
        filter_file = default_filter_file
    else:
        filter_file = user_input
        # 상대 경로인 경우 프로젝트 루트 기준으로 변환
        if not os.path.isabs(filter_file):
            filter_file = os.path.join(project_root, filter_file)
    
    if not os.path.exists(filter_file):
        print(f"\n✗ 파일을 찾을 수 없습니다: {filter_file}")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
        return
    
    # 2. 카페_결합문서.xlsx 파일 경로 입력
    print("\n[2단계] 카페_결합문서.xlsx 파일 경로를 입력하세요")
    print(f"기본값: {default_cafe_file}")
    print("엔터를 누르면 기본값을 사용합니다.")
    user_input = input("경로 입력: ").strip().replace('"', '')
    
    if not user_input:
        cafe_file = default_cafe_file
    else:
        cafe_file = user_input
        # 상대 경로인 경우 프로젝트 루트 기준으로 변환
        if not os.path.isabs(cafe_file):
            cafe_file = os.path.join(project_root, cafe_file)
    
    if not os.path.exists(cafe_file):
        print(f"\n✗ 파일을 찾을 수 없습니다: {cafe_file}")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
        return
    
    # 3. 저장 파일명 입력 (선택사항)
    print("\n[3단계] 저장할 파일명을 입력하세요")
    print(f"기본값: {default_output_file}")
    print("엔터를 누르면 기본값을 사용합니다.")
    user_input = input("경로 입력: ").strip().replace('"', '')
    
    if not user_input:
        output_file = default_output_file
    else:
        output_file = user_input
        # 상대 경로인 경우 프로젝트 루트 기준으로 변환
        if not os.path.isabs(output_file):
            output_file = os.path.join(project_root, output_file)
    
    if not output_file.endswith('.xlsx'):
        output_file += '.xlsx'
    
    if not output_file.endswith('.xlsx'):
        output_file += '.xlsx'
    
    print("\n" + "=" * 70)
    print("파일 처리 중...")
    print("=" * 70)
    
    try:
        # 4. 남은 행정동 파일 읽기
        print(f"\n✓ 필터 파일 로드: {filter_file}")
        df_filter = pd.read_excel(filter_file)
        
        # CD 컬럼 확인
        if 'CD' not in df_filter.columns:
            print(f"\n✗ 오류: 필터 파일에 'CD' 컬럼이 없습니다.")
            print(f"   발견된 컬럼: {df_filter.columns.tolist()}")
            return
        
        filter_codes = set(df_filter['CD'].astype(str).unique())
        print(f"  → 남은 행정동 개수: {len(filter_codes)}개")
        
        # 5. 카페 병합 파일 읽기
        print(f"\n✓ 카페 데이터 로드: {cafe_file}")
        df_cafe = pd.read_excel(cafe_file)
        print(f"  → 원본 데이터: {len(df_cafe):,}행 × {len(df_cafe.columns)}열")
        
        # CD 컬럼 확인
        if 'CD' not in df_cafe.columns:
            print(f"\n✗ 오류: 카페 파일에 'CD' 컬럼이 없습니다.")
            print(f"   발견된 컬럼: {df_cafe.columns.tolist()}")
            return
        
        # 6. CD 컬럼을 문자열로 변환하여 필터링
        df_cafe['CD'] = df_cafe['CD'].astype(str)
        df_filtered = df_cafe[df_cafe['CD'].isin(filter_codes)].copy()
        
        # 7. 결과 출력
        removed_count = len(df_cafe) - len(df_filtered)
        print(f"\n✓ 필터링 완료:")
        print(f"  → 유지된 데이터: {len(df_filtered):,}행")
        print(f"  → 제거된 데이터: {removed_count:,}행")
        print(f"  → 고유 행정동: {df_filtered['CD'].nunique()}개")
        
        # 8. 파일 저장
        df_filtered.to_excel(output_file, index=False)
        print(f"\n✓ 저장 완료: {os.path.abspath(output_file)}")
        
        print("\n" + "=" * 70)
        print("모든 작업이 완료되었습니다!")
        print("=" * 70)
        
        # 9. 샘플 데이터 출력
        print("\n[샘플 데이터 - 상위 5행]")
        if 'year' in df_filtered.columns and 'quarter' in df_filtered.columns:
            print(df_filtered[['year', 'quarter', 'CD', 'NM', 'GUBUN']].head())
        else:
            print(df_filtered.head())
            
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        filter_cafe_data()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
    
    input("\n\n엔터를 눌러 종료...")