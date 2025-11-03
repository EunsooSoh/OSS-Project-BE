import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sys
import os

# 로컬 timesfm 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import timesfm


def crawl_naver_samsung():
    """네이버 금융에서 삼성전자 주가 데이터를 크롤링합니다."""
    url = "https://finance.naver.com/item/sise_day.naver?code=005930"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print("네이버 금융에서 삼성전자 주가 데이터 크롤링 시작...")
    
    try:
        # pandas read_html을 사용한 간단한 방법
        from io import StringIO
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # pandas read_html로 테이블 직접 읽기
        tables = pd.read_html(StringIO(response.text))
        
        if not tables:
            raise ValueError("테이블을 찾을 수 없습니다.")
        
        # 첫 번째 테이블 사용
        df = tables[0]
        
        print(f"원본 데이터 shape: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        print("첫 5행:")
        print(df.head())
        
        # 결측값 제거
        df = df.dropna()
        
        # 컬럼 수에 따라 처리
        if len(df.columns) >= 6:
            # 일반적인 경우: 날짜, 종가, 전일비, 시가, 고가, 저가, 거래량
            df.columns = ['date', 'close', 'change', 'open', 'high', 'low', 'volume'][:len(df.columns)]
            df = df[['date', 'close', 'open', 'high', 'low']]
        elif len(df.columns) >= 4:
            # 컬럼이 적은 경우
            df.columns = ['date', 'close', 'open', 'volume'][:len(df.columns)]
            # 고가, 저가 추정
            df['high'] = df[['close', 'open']].max(axis=1)
            df['low'] = df[['close', 'open']].min(axis=1)
            df = df[['date', 'close', 'open', 'high', 'low']]
        else:
            raise ValueError("충분한 컬럼이 없습니다.")
        
        # 데이터 전처리
        # 날짜 처리
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # 숫자 컬럼 처리
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('-', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 결측값 제거
        df = df.dropna()
        
        # 날짜순 정렬
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"전처리 완료: {len(df)}개 데이터")
        print("최종 데이터:")
        print(df.head())
        
        if len(df) == 0:
            raise ValueError("전처리 후 데이터가 없습니다.")
        
        return df
        
    except Exception as e:
        print(f"크롤링 오류: {e}")
        raise ValueError(f"크롤링 실패: {e}")


def predict_with_timesfm_week(df):
    """TimesFM을 사용하여 7일간의 주가를 예측합니다."""
    # 7일 예측
    series = df["close"].values.astype(np.float32)
    series = (series - series.min()) / (series.max() - series.min())
    
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=7,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))
    
    forecast_horizon = 7
    point_forecast, quantile_forecast = model.forecast(
        horizon=forecast_horizon, 
        inputs=[series]
    )
    
    # 역정규화
    min_val, max_val = df["close"].min(), df["close"].max()
    pred = point_forecast[0] * (max_val - min_val) + min_val
    q10 = quantile_forecast[0, :, 1] * (max_val - min_val) + min_val
    q90 = quantile_forecast[0, :, 9] * (max_val - min_val) + min_val
    
    return pred, q10, q90


if __name__ == "__main__":
    # 데이터를 크롤링 후 예측 수행
    df = crawl_naver_samsung()
    pred_week, q10_week, q90_week = predict_with_timesfm_week(df)
    
    print("다음 일주일 예측:", pred_week)
    print("Q10 (상한):", q10_week)
    print("Q90 (하한):", q90_week)