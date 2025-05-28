# 📦 라이브러리 임포트
import yfinance as yf
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from pandas_datareader import data as web
from datetime import datetime
import pytz
import os
from bs4 import BeautifulSoup

# ✅ PER/PBR 캐시 (네이버용)
CACHE_FILE = "per_pbr_cache.csv"
def save_to_cache(ticker, per, pbr):
    try:
        df = pd.read_csv(CACHE_FILE) if os.path.exists(CACHE_FILE) else pd.DataFrame(columns=["ticker", "per", "pbr"])
        df = df[df["ticker"] != ticker]
        df = pd.concat([df, pd.DataFrame([[ticker, per, pbr]], columns=["ticker", "per", "pbr"])], ignore_index=True)
        df.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
    except: pass

def load_from_cache(ticker):
    try:
        if not os.path.exists(CACHE_FILE): return "수집 실패", "수집 실패"
        df = pd.read_csv(CACHE_FILE)
        row = df[df["ticker"] == ticker]
        return row.iloc[0]["per"], row.iloc[0]["pbr"] if not row.empty else ("수집 실패", "수집 실패")
    except:
        return "수집 실패", "수집 실패"

# ✅ 해석 함수들
def interpret_per(val):
    try: val = float(str(val).replace(',', '')); return "매수" if val < 10 else "중립" if val <= 25 else "매도"
    except: return "해석불가"

def interpret_pbr(val):
    try: val = float(str(val).replace(',', '')); return "매수" if val < 1 else "중립" if val <= 3 else "매도"
    except: return "해석불가"

def interpret_ratio(val, low, high):
    try: val = float(val); return "매수" if val < low else "중립" if val <= high else "매도"
    except: return "해석불가"

def interpret_rsi(val):
    try: return "매수" if val < 30 else "중립" if val <= 70 else "매도"
    except: return "해석불가"

def interpret_moving_avg(current, ma):
    try:
        current, ma = float(current), float(ma)
        diff = (current - ma) / ma * 100
        return f"{ma:.2f} (매수)" if diff > 1 else f"{ma:.2f} (매도)" if diff < -1 else f"{ma:.2f} (중립)"
    except: return "수집 실패"

def interpret_dividend_yield(val):
    try: val = float(val); val *= 100 if val < 1 else 1; return "매수" if val >= 3 else "중립" if val >= 1 else "매도"
    except: return "해석불가"

def interpret_profit_margin(val):
    try: val = float(val); val *= 100 if val < 1 else 1; return "매수" if val >= 15 else "중립" if val >= 5 else "매도"
    except: return "해석불가"

def interpret_forward_pe(val):
    try: val = float(val); return "매수" if val < 10 else "중립" if val <= 25 else "매도"
    except: return "해석불가"

def interpret_52w_change(val):
    try: val = float(val); val *= 100 if abs(val) < 1 else 1; return "매수" if val >= 20 else "중립" if val >= -10 else "매도"
    except: return "해석불가"

def format_market_cap(val):
    try:
        val = int(val)
        return f"{val / 1_0000_0000_0000:.1f}조" if val >= 1_0000_0000_0000 else \
               f"{val / 1_0000_0000:.0f}억" if val >= 1_0000_0000 else \
               f"{val / 1_0000:.0f}만" if val >= 1_0000 else str(val)
    except: return "수집 실패"

# ✅ RSI 계산
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)).iloc[-1], 2)

def get_per_pbr_naver(ticker):
    code = ticker.split('.')[0]
    url = f"https://finance.naver.com/item/main.naver?code={code}"

    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        res.raise_for_status()

        soup = BeautifulSoup(res.text, 'html.parser')
        per_tag = soup.select_one('em#_per')
        pbr_tag = soup.select_one('em#_pbr')

        per = per_tag.text.strip() if per_tag else "수집 실패"
        pbr = pbr_tag.text.strip() if pbr_tag else "수집 실패"

        save_to_cache(ticker, per, pbr)
        return per, pbr
    except:
        return load_from_cache(ticker)

# ✅ 패딩 함수 (거시지표용)
def pad_df(df, target_cols):
    pad_cols = len(target_cols) - df.shape[1]
    for i in range(pad_cols): df[f"빈칸{i+1}"] = ""
    return df

# ✅ 주식 리스트 (샘플)
kr_stocks = {    '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '373220.KS': 'LG에너지솔루션',
    '207940.KS': '삼성바이오로직스', '005935.KS': '삼성물산', '005380.KS': '현대차',
    '000270.KS': '기아', '005490.KS': 'POSCO홀딩스', '035420.KS': 'LG화학', '035720.KS': 'NAVER',
    '051910.KS': '삼성SDI', '012330.KS': '현대모비스', '066570.KS': 'LG전자', '055550.KS': '신한지주',
    '105560.KS': 'KB금융', '086790.KS': '하나금융지주', '032640.KS': '삼성생명', '015760.KS': '한국전력',
    '003490.KS': '대한항공', '011780.KS': 'HMM', '096770.KS': 'SK이노베이션', '005180.KS': 'KT&G',
    '010950.KS': 'S-Oil', '032830.KS': '우리금융지주', '009150.KS': '삼성전기', '259960.KS': '크래프톤',
    '251270.KS': '넷마블', '036570.KS': '엔씨소프트', '000220.KS': '고려아연', '090430.KS': '아모레퍼시픽',
    '034220.KS': 'LG디스플레이', '000240.KS': '한국타이어', '042670.KS': '한국항공우주', '010120.KS': 'LS ELECTRIC',
    '019570.KS': '한미사이언스', '024110.KS': 'SK', '007070.KS': 'GS리테일', '139480.KS': '이마트',
    '011170.KS': '롯데케미칼', '004020.KS': '한화', '067010.KS': '롯데쇼핑', '006800.KS': '삼천리',
    '068270.KS': '셀트리온', '033780.KS': '한국가스공사', '000810.KS': '신세계', '007310.KS': '대우건설',
    '086280.KS': '삼성중공업', '003030.KS': '한국항공우주', '000720.KS': 'SK네트웍스',
    '023530.KS': '아모레퍼시픽', '003060.KS': '대우조선해양', '027410.KS': '한화에어로스페이스'}
us_stocks = {'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOG': 'Alphabet', 'AMZN': 'Amazon', 'META': 'Meta Platforms',
    'TSLA': 'Tesla', 'BRK-B': 'Berkshire Hathaway', 'V': 'Visa', 'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth',
    'XOM': 'Exxon Mobil', 'JPM': 'JPMorgan Chase', 'PG': 'Procter & Gamble', 'MA': 'Mastercard', 'HD': 'Home Depot',
    'CVX': 'Chevron', 'LLY': 'Eli Lilly', 'PFE': 'Pfizer', 'KO': 'Coca-Cola', 'PEP': 'PepsiCo', 'MRK': 'Merck',
    'ABBV': 'AbbVie', 'WMT': 'Walmart', 'INTC': 'Intel', 'CSCO': 'Cisco', 'CMCSA': 'Comcast', 'QCOM': 'Qualcomm',
    'AVGO': 'Broadcom', 'TMO': 'Thermo Fisher', 'NKE': 'Nike', 'ORCL': 'Oracle', 'TXN': 'Texas Instruments',
    'IBM': 'IBM', 'CRM': 'Salesforce', 'ADBE': 'Adobe', 'COST': 'Costco', 'AMGN': 'Amgen', 'INTU': 'Intuit',
    'NFLX': 'Netflix', 'AMD': 'AMD', 'SBUX': 'Starbucks', 'GILD': 'Gilead Sciences', 'BKNG': 'Booking Holdings',
    'PYPL': 'PayPal', 'GE': 'General Electric', 'MMM': '3M', 'CVS': 'CVS Health', 'F': 'Ford', 'MRNA': 'Moderna'}
all_stocks = {**kr_stocks, **us_stocks}

# ✅ 주식 데이터 수집
results = []
for ticker, name in all_stocks.items():
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")
        h2 = stock.history(period="5d")
        if hist.empty or len(h2) < 2: continue
        today = h2['Close'].iloc[-1]
        prev = h2['Close'].iloc[-2]
        change = today - prev
        rate = round(change / prev * 100, 2)

        is_kr = ticker.endswith('.KS') or ticker.endswith('.KQ')
        per, pbr = get_per_pbr_naver(ticker) if is_kr else (
            info.get("trailingPE", "수집 실패"),
            info.get("priceToBook", "수집 실패")
        )

        roe = info.get("returnOnEquity", "수집 실패")
        debt = info.get("debtToEquity", "수집 실패")
        div = info.get("dividendYield", "수집 실패")
        margin = info.get("profitMargins", "수집 실패")
        fpe = info.get("forwardPE", "수집 실패")
        chg52 = info.get("52WeekChange", "수집 실패")
        volume = info.get("volume", "수집 실패")
        mcap = format_market_cap(info.get("marketCap", "수집 실패"))
        rsi = calculate_rsi(hist["Close"])
        ma5 = round(hist['Close'].rolling(5).mean().iloc[-1], 2)
        ma20 = round(hist['Close'].rolling(20).mean().iloc[-1], 2)

        results.append({
            "종목명": name, "티커": ticker,
            "당일종가": round(today, 2), "전날종가": round(prev, 2), "변화량": round(change, 2), "변화율(%)": rate,
            "PER": per, "PER_해석": interpret_per(per),
            "PBR": pbr, "PBR_해석": interpret_pbr(pbr),
            "ROE": roe, "ROE_해석": interpret_ratio(roe, 0.1, 0.2),
            "Debt/Equity": debt, "Debt/Equity_해석": interpret_ratio(debt, 50, 150),
            "Dividend Yield": div, "배당률_해석": interpret_dividend_yield(div),
            "Profit Margin": margin, "이익률_해석": interpret_profit_margin(margin),
            "Forward PE": fpe, "Forward PE_해석": interpret_forward_pe(fpe),
            "52주 변화율": chg52, "52주 해석": interpret_52w_change(chg52),
            "RSI": rsi, "RSI_해석": interpret_rsi(rsi),
            "5일 이평선": interpret_moving_avg(today, ma5),
            "20일 이평선": interpret_moving_avg(today, ma20),
            "거래량": volume, "시가총액": mcap
        })

    except Exception as e:
        print(f"❌ {name} 에러 발생: {e}")

stock_df = pd.DataFrame(results)

# ✅ 미국 거시지표
def get_us_macro_df_and_print():
    us_indicators = {
        'GDP': '국내총생산', 'FEDFUNDS': '연방기금금리', 'UNRATE': '실업률', 'CPIAUCNS': '소비자물가지수',
        'CPILFESL': '근원 CPI', 'PCE': '개인소비지출', 'UMCSENT': '소비자심리지수',
        'ICSA': '신규 실업수당청구건수', 'GS10': '10년 국채 수익률', 'USSLIND': '경기 선행지수'
    }
    print("\n📊 [미국 거시지표 Top 10 - FRED]\n")
    rows = []
    for code, name in us_indicators.items():
        try:
            df = web.DataReader(code, 'fred', '2010-01-01', datetime.today()).dropna()
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            latest_value = df.iloc[-1, 0]
            print(f"🔹 {name} [{latest_date}]: {latest_value}")
            rows.append([name, code, latest_date, latest_value])
        except Exception as e:
            print(f"❌ {name} 수집 실패: {e}")
            rows.append([name, code, "수집 실패", "수집 실패"])
    return pd.DataFrame(rows, columns=["지표명", "코드", "날짜", "값"])

# ✅ 한국 거시지표
def get_korea_macro_df_and_print():
    print("\n📊 [한국 거시지표 - ECOS 주요지표 10선]\n")
    API_KEY = 'VADVWXGUAJ1D7O7H9PKX'
    url = f'https://ecos.bok.or.kr/api/KeyStatisticList/{API_KEY}/xml/kr/1/100'
    res = requests.get(url)
    root = ET.fromstring(res.content)

    keywords = ["기준금리", "콜금리", "수신금리", "대출금리", "가계신용", "M2", "환율(종가)", "GDP", "소비자물가지수", "실업률"]

    rows = []
    idx = 1
    for row in root.findall('.//row'):
        title = row.findtext('KEYSTAT_NAME', default="N/A").strip()
        if any(k in title for k in keywords):
            time = row.findtext('TIME', default="N/A").strip()
            value = row.findtext('DATA_VALUE', default="N/A").strip()
            unit = row.findtext('UNIT_NAME', default="").strip()
            print(f"{idx:>2}. {title} [{time}]: {value} {unit}")
            rows.append([title, time, value, unit])
            idx += 1
        if idx > 10: break
    return pd.DataFrame(rows, columns=["지표명", "날짜", "값", "단위"])

# ✅ CSV 파일 저장
filename = "data/stock_price.csv"

us_df = get_us_macro_df_and_print()
kr_df = get_korea_macro_df_and_print()

blank = pd.DataFrame([[""] * stock_df.shape[1]], columns=stock_df.columns)
us_title = pd.DataFrame([["[미국 거시지표]"] + [""] * (stock_df.shape[1] - 1)], columns=stock_df.columns)
kr_title = pd.DataFrame([["[한국 거시지표]"] + [""] * (stock_df.shape[1] - 1)], columns=stock_df.columns)
us_df_padded = pad_df(us_df.copy(), stock_df.columns)
kr_df_padded = pad_df(kr_df.copy(), stock_df.columns)

final_df = pd.concat([stock_df, blank, us_title, us_df_padded, blank, kr_title, kr_df_padded], ignore_index=True)
final_df = final_df.dropna(axis=1, how='all')  # ⚠️ 여기서 'how="any"' 대신 'how="all"'도 고려 가능
os.makedirs(os.path.dirname(filename), exist_ok=True)
final_df.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"\n✅ 최종 CSV 저장 완료 → {filename}")

# ✅ 엑셀 파일 미리보기 (하위 20줄)
# ✅ 모든 열 출력 설정 (열 생략 방지)
pd.set_option('display.max_columns', None)

# ✅ CSV 저장 후 미리보기 (NaN 포함, 줄바꿈 없이 전체 출력)
print("\n📄 [미리보기 - 상위 2줄]")
print(final_df.tail(20).to_string(index=False))
