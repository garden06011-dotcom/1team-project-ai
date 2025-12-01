# #################################
# 서울 골목상권분석 데이터 수집 스크립트 - 정원 크롤링 코드
# #################################

"""
서울 골목상권분석 데이터 수집 (원본 데이터 그대로 저장)
https://golmok.seoul.go.kr/stateArea.do

API에서 받은 JSON 데이터를 가공하지 않고 엑셀에 저장
"""

import requests
import pandas as pd
import time
from datetime import datetime
import json

class GolmokDataCollector:
    def __init__(self):
        """데이터 수집기 초기화"""
        self.base_url = "https://golmok.seoul.go.kr"
        self.session = requests.Session()
        
        # HTTP 헤더 설정
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'https://golmok.seoul.go.kr',
            'Referer': 'https://golmok.seoul.go.kr/stateArea.do'
        })
        
        # 수집 설정
        self.years = ['2020', '2021', '2022', '2023', '2024']
        self.quarters = ['1', '2', '3', '4']
        self.industries = {
            '커피음료': 'CS100001',
            '한식음식점': 'CS100009',
            '호프간이주점': 'CS100010'
        }
        
        # API 엔드포인트
        self.endpoints = {
            '점포수': '/region/selectStoreCount.json',
            '신생기업생존율': '/region/selectQuaterData.json',
            '연차별생존율': '/region/selectYearData.json',
            '평균영업기간': '/region/selectMonthData.json',
            '개폐업수': '/region/selectOpening.json',
            '인구수': '/region/selectPopulation.json',
            '소득가구수': '/region/selectIncome.json',
            '임대시세': '/region/selectRentalPrice.json'
        }
        
        # 수집된 데이터 저장
        self.collected_data = {
            '점포수': [],
            '신생기업생존율': [],
            '연차별생존율': [],
            '평균영업기간': [],
            '개폐업수': [],
            '인구수': [],
            '소득가구수': [],
            '임대시세': []
        }
    
    def create_form_data(self, year, quarter, industry_code):
        """
        API 요청용 Form Data 생성
        """
        quarter_month = {
            '1': '03',
            '2': '06',
            '3': '09',
            '4': '12'
        }
        stdr_mn_cd = f"{year}{quarter_month[quarter]}"
        
        form_data = {
            'stdrYyCd': year,
            'stdrSlctQu': 'sameQu',
            'stdrQuCd': quarter,
            'stdrMnCd': stdr_mn_cd,
            'selectTerm': 'quarter',
            'svcIndutyCdL': industry_code,
            'svcIndutyCdM': industry_code,
            'stdrSigngu': '11',
            'selectInduty': '1',
            'infoCategory': 'store'
        }
        
        return form_data
    
    def fetch_data(self, endpoint, year, quarter, industry_code):
        """API 데이터 가져오기"""
        try:
            url = f"{self.base_url}{endpoint}"
            form_data = self.create_form_data(year, quarter, industry_code)
            
            response = self.session.post(url, data=form_data)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return None
    
    def collect_all_data(self):
        """모든 데이터 수집"""
        print("="*70)
        print("서울 골목상권분석 데이터 수집 시작")
        print("="*70)
        print()
        
        total_requests = len(self.years) * len(self.quarters) * len(self.industries) * len(self.endpoints)
        current = 0
        
        for year in self.years:
            for quarter in self.quarters:
                for industry_name, industry_code in self.industries.items():
                    print(f"\n📍 {year}년 {quarter}분기 - {industry_name}")
                    
                    for data_name, endpoint in self.endpoints.items():
                        current += 1
                        progress = (current / total_requests) * 100
                        print(f"  [{progress:5.1f}%] {data_name} 수집 중...", end=' ')
                        
                        # API 호출
                        json_data = self.fetch_data(endpoint, year, quarter, industry_code)
                        
                        if json_data:
                            # 원본 데이터에 메타정보 추가
                            if isinstance(json_data, list):
                                # 리스트 형태인 경우 각 항목에 메타정보 추가
                                for item in json_data:
                                    item['_연도'] = year
                                    item['_분기'] = quarter
                                    item['_업종'] = industry_name
                                    item['_업종코드'] = industry_code
                                    self.collected_data[data_name].append(item)
                            elif isinstance(json_data, dict):
                                # 딕셔너리 형태인 경우 메타정보 추가
                                json_data['_연도'] = year
                                json_data['_분기'] = quarter
                                json_data['_업종'] = industry_name
                                json_data['_업종코드'] = industry_code
                                self.collected_data[data_name].append(json_data)
                            
                            print("✓")
                        else:
                            print("✗")
                        
                        time.sleep(0.3)  # API 부하 방지
        
        print()
        print("="*70)
        print("데이터 수집 완료!")
        print("="*70)
        
        # 수집 결과 요약
        print("\n📊 수집 결과:")
        for data_name, data_list in self.collected_data.items():
            print(f"  - {data_name}: {len(data_list)}건")
    
    def save_to_excel(self, filename='골목상권분석_원본데이터.xlsx'):
        """엑셀 파일로 저장"""
        print(f"\n💾 엑셀 파일 저장 중: {filename}")
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                for data_name, data_list in self.collected_data.items():
                    if not data_list:
                        print(f"  ⚠ {data_name}: 데이터 없음")
                        continue
                    
                    # DataFrame 생성
                    df = pd.DataFrame(data_list)
                    
                    # 메타정보 컬럼을 앞으로 이동
                    meta_cols = ['_연도', '_분기', '_업종', '_업종코드']
                    other_cols = [col for col in df.columns if col not in meta_cols]
                    df = df[meta_cols + other_cols]
                    
                    # 시트명 (최대 31자)
                    sheet_name = data_name[:31]
                    
                    # 엑셀에 저장
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✓ {sheet_name}: {len(df)}행 저장")
            
            print(f"\n✅ 저장 완료: {filename}")
            return True
            
        except Exception as e:
            print(f"\n❌ 저장 실패: {e}")
            return False


def main():
    """메인 실행"""
    print()
    print("="*70)
    print("  서울 골목상권분석 데이터 수집 프로그램")
    print("="*70)
    print()
    print("📋 수집 설정:")
    print("  - 기간: 2020~2024년 (각 1~4분기)")
    print("  - 업종: 커피음료, 한식음식점, 호프간이주점")
    print("  - 항목: 점포수, 신생기업생존율, 연차별생존율, 평균영업기간,")
    print("          개폐업수, 인구수, 소득가구수, 임대시세")
    print()
    print("⏱️  예상 소요 시간: 약 5~7분")
    print()
    
    input("계속하려면 Enter를 누르세요...")
    print()
    
    # 데이터 수집
    collector = GolmokDataCollector()
    collector.collect_all_data()
    
    # 엑셀 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'골목상권분석_데이터_{timestamp}.xlsx'
    
    if collector.save_to_excel(filename):
        print()
        print("="*70)
        print("🎉 작업 완료!")
        print(f"📁 파일: {filename}")
        print("="*70)
    else:
        print("\n❌ 작업 실패")


if __name__ == "__main__":
    main()