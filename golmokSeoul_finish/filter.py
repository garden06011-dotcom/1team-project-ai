import pandas as pd
from tkinter import Tk, filedialog

def filter_cafe_data():
    """
    카페 병합 파일에서 남은 행정동에 있는 CD만 남기고 나머지 삭제
    """
    # Tkinter 루트 윈도우 생성 (숨김)
    root = Tk()
    root.withdraw()
    
    print("=" * 60)
    print("카페 데이터 필터링 프로그램")
    print("=" * 60)
    
    # 1. 남은_행정동2.xlsx 파일 선택
    print("\n[1단계] 남은_행정동2.xlsx 파일을 선택하세요...")
    filter_file = filedialog.askopenfilename(
        title="남은_행정동2.xlsx 파일 선택",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    
    if not filter_file:
        print("파일 선택이 취소되었습니다.")
        return
    
    # 2. 카페_결합문서.xlsx 파일 선택
    print("\n[2단계] 카페_결합문서.xlsx 파일을 선택하세요...")
    cafe_file = filedialog.askopenfilename(
        title="카페_결합문서.xlsx 파일 선택",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    
    if not cafe_file:
        print("파일 선택이 취소되었습니다.")
        return
    
    # 3. 저장 위치 선택
    print("\n[3단계] 필터링된 파일을 저장할 위치를 선택하세요...")
    output_file = filedialog.asksaveasfilename(
        title="저장할 파일명 입력",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialfile="카페_결합문서_필터링.xlsx"
    )
    
    if not output_file:
        print("저장 위치 선택이 취소되었습니다.")
        return
    
    print("\n" + "=" * 60)
    print("파일 처리 중...")
    print("=" * 60)
    
    # 4. 남은 행정동 파일 읽기
    print(f"\n✓ 필터 파일 로드: {filter_file}")
    df_filter = pd.read_excel(filter_file)
    filter_codes = set(df_filter['CD'].astype(str).unique())
    print(f"  → 남은 행정동 개수: {len(filter_codes)}개")
    
    # 5. 카페 병합 파일 읽기
    print(f"\n✓ 카페 데이터 로드: {cafe_file}")
    df_cafe = pd.read_excel(cafe_file)
    print(f"  → 원본 데이터: {len(df_cafe):,}행 × {len(df_cafe.columns)}열")
    
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
    print(f"\n✓ 저장 완료: {output_file}")
    
    print("\n" + "=" * 60)
    print("모든 작업이 완료되었습니다!")
    print("=" * 60)
    
    # 9. 샘플 데이터 출력
    print("\n[샘플 데이터 - 상위 5행]")
    print(df_filtered.head())

if __name__ == "__main__":
    filter_cafe_data()