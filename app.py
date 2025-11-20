from flask import Flask, request, jsonify
import joblib
from pyproj import Transformer
import numpy as np
import pandas as pd




app = Flask(__name__)


SCORE_WEIGHT = {
    '생활인구': 0.4,
    '임대료': 0.3,
    '업체수': 0.3,
}

#유동인구 관련 모델 및 데이터
human_loaded = joblib.load("models/human_model.pkl")
human_model=human_loaded['model']
human_scaler=human_loaded['scaler']
human_feature_cols=human_loaded['feature_cols']


#임대료
rent_loaded = joblib.load("models/rent_model_real.pkl")
rent_model=rent_loaded['model']
rent_scaler=rent_loaded['scaler']
rent_feature_cols=rent_loaded['feature_cols']


#경쟁업체, 카페, 한식, 호프
cafe_model = joblib.load('models/store_model_cafe.pkl')
hansic_model = joblib.load('models/store_model_hansic.pkl')
hope_model = joblib.load('models/store_model_hope.pkl')

dict_model = {
    'cafe': {
        'model': cafe_model['model'],
        'scaler': cafe_model['scaler'],
        'feature_cols': cafe_model['feature_cols']
    },
    'hansic': {
        'model': hansic_model['model'],
        'scaler': hansic_model['scaler'],
        'feature_cols': hansic_model['feature_cols']
    },
    'hope': {
        'model': hope_model['model'],
        'scaler': hope_model['scaler'],
        'feature_cols': hope_model['feature_cols']
    }
}


transformer = Transformer.from_crs("epsg:4326", "epsg:5179", always_xy=True)

@app.route("/predict", methods=['POST'])
def predict():
    #post request body에서 x, y 좌표 받기
    data = request.get_json()
    lng = float(data['lng']) #경도
    lat = float(data['lat']) #위도
    
    # 변환 전 좌표 출력
    print(f"입력 좌표 (경위도): lng={lng}, lat={lat}")
    
    # 좌표 변환 (EPSG:4326 -> EPSG:5179)
    try:
        x, y = transformer.transform(lng, lat)
        
        # 변환 결과 검증
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise ValueError(f"변환 결과가 숫자가 아닙니다: x={x}, y={y}")
        
        if not (abs(x) < 1e10 and abs(y) < 1e10):  # inf나 너무 큰 값 체크
            raise ValueError(f"변환 결과가 유효하지 않습니다: x={x}, y={y}")
        
        print(f"변환된 좌표 (직교좌표계): x={x}, y={y}")


        # 유동인구 예측
        human_coords_scaled = human_scaler.transform([[x, y]])[0]
        human_new_x_scaled = human_coords_scaled[0]
        human_new_y_scaled = human_coords_scaled[1]

        human_new_features = [[
            human_new_x_scaled, 
            human_new_y_scaled, 
            human_new_x_scaled**2, 
            human_new_y_scaled**2, 
            human_new_x_scaled * human_new_y_scaled
        ]]

        human_pred = human_model.predict(human_new_features)

        # 임대료 예측
        rent_coords_scaled = rent_scaler.transform([[x, y]])[0]
        rent_new_x_scaled = rent_coords_scaled[0]
        rent_new_y_scaled = rent_coords_scaled[1]

        
        # feature_cols에 따라 동적으로 특징 생성
        feature_dict = {
            'x_meter': rent_new_x_scaled,
            'y_meter': rent_new_y_scaled,
            'x_meter_squared': rent_new_x_scaled**2,
            'y_meter_squared': rent_new_y_scaled**2,
            'x_meter_y_meter': rent_new_x_scaled * rent_new_y_scaled
        }

        # feature_cols의 순서에 맞게 특징 배열 생성
        rent_new_features = []
        for col in rent_feature_cols:
            if col in feature_dict:
                rent_new_features.append(feature_dict[col])
            else:
                # 추가 특징이 있는 경우 0으로 채우거나 적절한 기본값 사용
                # (실제로는 해당 특징의 의미에 맞는 값을 제공해야 함)
                print(f"경고: '{col}' 특징이 없어 0으로 채웁니다.")
                rent_new_features.append(0.0)

        rent_new_features = [rent_new_features]

      

        rent_pred = rent_model.predict(rent_new_features)

        result_dict = {}

        #카페,한식,호프 예측
        for key, value in dict_model.items():
            model = value['model']
            scaler = value['scaler']
            feature_cols = value['feature_cols']
            
            new_coords_scaled = scaler.transform([[x, y]])[0]
            new_x_scaled = new_coords_scaled[0]
            new_y_scaled = new_coords_scaled[1]

            new_features = [[
                new_x_scaled, 
                new_y_scaled, 
                new_x_scaled**2, 
                new_y_scaled**2, 
                new_x_scaled * new_y_scaled
            ]]

            new_pred = model.predict(new_features)
            result_dict[key] = float(new_pred[0])

        
        score = sum(SCORE_WEIGHT.values()) * (float(human_pred[0]) * SCORE_WEIGHT['생활인구'] + float(rent_pred[0]/10) * SCORE_WEIGHT['임대료'] + float(result_dict['cafe']) * SCORE_WEIGHT['업체수'] + float(result_dict['hansic']) * SCORE_WEIGHT['업체수'] + float(result_dict['hope']) * SCORE_WEIGHT['업체수'])
        
        return jsonify({
            '생활인구': float(human_pred[0]), 
            '임대료': float(rent_pred[0]/10), 
            '카페경젱업체수': float(result_dict['cafe']),
            '음식점경쟁업체수': float(result_dict['hansic']),
            '호프경쟁업체수': float(result_dict['hope']),
            '점수' : round(score, 2),   
            '응답설명':'생활인구단위 : (명), 임대료단위 : (만원/제곱미터), 경쟁업체수 단위 : (개)'
        })
    
    except Exception as e:
        
        return jsonify({'error': f'오류: {str(e)}'}), 400

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
