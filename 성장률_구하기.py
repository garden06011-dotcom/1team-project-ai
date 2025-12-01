import pandas as pd
import numpy as np

# 1. 엑셀 불러오기
file_path = 'golmokSeoul_finish/data/1130_호프_데이터.xlsx' 
df = pd.read_excel(file_path)

# 2. 성장률 계산
# 조건: 작년 매출이 NaN이 아니고, 0이 아닌 경우만 계산
df['성장률'] = df.apply(
    lambda row: (row['매출'] - row['작년 매출']) / row['작년 매출']
    if pd.notna(row['작년 매출']) and row['작년 매출'] != 0
    else np.nan,
    axis=1
)

# 3. 엑셀 저장
output_path = '매출데이터_호프_성장률추가.xlsx'
df.to_excel(output_path, index=False)

print("📌 완료! 저장된 파일:", output_path)
