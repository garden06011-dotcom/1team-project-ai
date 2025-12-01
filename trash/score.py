import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import joblib

# ============================================================
# 1) 데이터 로드
# ============================================================
df = pd.read_excel("final-data.xlsx")

# ============================================================
# 2) 점포수 → 경쟁도 역전환 처리
# ============================================================
df["전체_역점포"] = 1 / (df["전체점포수"] + 1)
df["음식점_역점포"] = 1 / (df["음식점점포수"] + 1)
df["카페_역점포"]   = 1 / (df["카페점포수"] + 1)
df["호프_역점포"]   = 1 / (df["호프점포수"] + 1)

# PCA 변수
X_cols = [
    "길단위유동인구","주거인구",
    "전체_역점포","음식점_역점포","카페_역점포","호프_역점포",
    "당월매출","주말매출","주중매출"
]

# ============================================================
# 3) 연도 가중치 (엑셀에는 표시 안 함)
# ============================================================
year_weights = {
    2020: 0.2,
    2022: 0.3,
    2024: 0.5
}

# ============================================================
# 4) 구 단위 PCA 수행 + loadings 저장 + percentile (0~100)
# ============================================================
results = []

for gu, g in df.groupby("행정구"):
    temp = g.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(temp[X_cols])

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).flatten()
    temp["PC1_raw"] = pc1

    # PCA 방향 자동 보정 (유동인구와 같은 방향)
    corr = np.corrcoef(temp["PC1_raw"], temp["길단위유동인구"])[0, 1]
    if corr < 0:
        temp["PC1_raw"] *= -1
        pca.components_[0] *= -1

    # percentile → 0~100
    percentile = rankdata(temp["PC1_raw"], method="min") / len(temp)
    temp["입지점수_0_100"] = percentile * 100

    # PCA loadings 저장
    loadings = pd.Series(pca.components_[0], index=X_cols)
    for col, w in loadings.items():
        temp[f"PCA_weight_{col}"] = w

    results.append(temp)

df_pca_full = pd.concat(results).reset_index(drop=True)

# ============================================================
# 5) 동(행정동) 단위 → 연도 가중치 반영하여 최종 점수 계산
# ============================================================
final_scores = []

for (gu, dong), g in df_pca_full.groupby(["행정구", "행정동"]):

    score_sum = 0
    weight_sum = 0

    for year, w in year_weights.items():
        g_year = g[g["년도"] == year]
        if len(g_year) > 0:
            yearly_avg = g_year["입지점수_0_100"].mean()
            score_sum += yearly_avg * w
            weight_sum += w

    final_score = score_sum / weight_sum if weight_sum > 0 else 0

    final_scores.append({
        "행정구": gu,
        "행정동": dong,
        "최종입지점수(0~100)": final_score
    })

df_final = pd.DataFrame(final_scores)

# ============================================================
# 6) 엑셀 + PKL 저장
# ============================================================
df_pca_full.to_excel("score_weights.xlsx", index=False)
df_final.to_excel("dong_score.xlsx", index=False)

joblib.dump({
    "X_cols": X_cols,
    "year_weights": year_weights,
    "pca_full": df_pca_full,
    "dong_score": df_final
}, "dong_score.pkl")

print("✔ score_weights.xlsx 생성 완료")
print("✔ dong_score.xlsx 생성 완료")
print("✔ dong_score.pkl 저장 완료")
