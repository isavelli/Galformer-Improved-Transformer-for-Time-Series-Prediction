import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import zscore

START_DATE = '2005-01-01'
FOLDER = 'data'

def get_sp500_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]
    return df['Symbol'].str.replace('.', '-').tolist()

def get_russell1000_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/Russell_1000_Index')[3]
    return df['Symbol'].str.replace('.', '-').tolist()

def get_dax40_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/DAX')[4]
    return df['Ticker'].tolist()

def get_cac40_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/CAC_40')[4]
    return df['Ticker'].tolist()

def get_ftsemib_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/FTSE_MIB')[1]
    return df['Ticker'].tolist()

def get_ftseItaliaSTAR_tickers():
    df = pd.read_html('https://it.wikipedia.org/wiki/FTSE_Italia_STAR')[0]
    df['Sigla'] += '.MI'
    return df['Sigla'].tolist()

def get_ftse100_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index')[6]
    df['Ticker'] += '.L'
    return df['Ticker'].tolist()

def get_ibex35_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/IBEX_35')[2]
    return df['Ticker'].tolist()

def get_euronext100_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/Euronext_100')[3]
    list = df['Ticker'].tolist()
    list.remove('SHEEL.AS')
    return list

def get_smi_tickers():
    df = pd.read_html('https://en.wikipedia.org/wiki/Swiss_Market_Index')[2]
    return df['Ticker'].tolist()


def download_ohlcv_data(tickers, start=None, end=None, auto_adjust=True, progress=False, period="max", interval="1d", multi_level_index=False, check_folder=None):
    data = {}
    for ticker in tickers:
        if check_folder is not None:
            file_path = f"{check_folder}/{ticker}.csv"
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                continue
        try:
            print(f"Downloading {ticker}")

            df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=progress, period=period, interval=interval, multi_level_index=multi_level_index)

            if df.empty:
                raise Exception(f'The dataframe is empty')

            if df.isna().any().any():
                raise Exception(f'Detected NA values')

            running_start = start
            while (df < 0).any().any():
                running_start = pd.to_datetime(running_start) + pd.DateOffset(months=6)
                print(f'Detected negative values, new start date: {running_start}')
                df = yf.download(ticker, start=running_start, end=end, auto_adjust=auto_adjust, progress=progress, period=period, interval=interval, multi_level_index=multi_level_index)

            # Volume can also start with zero (e.g. ^FCHI) and removing also those lines significantly
            # reduces the data available, therefore Volume has been removed
            # df['Volume'] = df['Volume'].replace(0, pd.NA).ffill()
            df.drop(columns='Volume', inplace=True)

            if (df <= 0).any().any():
                raise Exception(f'Detected non-strictly positive values')

            # # Detect outliers
            # df_prev = df[['Open', 'High', 'Low', 'Close']].shift(1)  # Shift Close column to get previous day's close
            # df_perc = df[['Open', 'High', 'Low', 'Close']] / df_prev - 1  # Calculate percentage change with respect to previous close
            # df_perc = df_perc.dropna()     # Drop first row (or any row) with NaN
            #
            # z_scores = df_perc.apply(zscore)  # Compute z-scores per column
            #
            # outliers_mask = (z_scores.abs() > 30)  # beyond 30 st.dev.
            # if outliers_mask.any().any():
            #     outlier_values = df_perc[outliers_mask]
            #     raise Exception(f'z-score detected possible outliers \n{outlier_values.dropna(how="all")}\n---------')

            data[ticker] = df[['Open', 'High', 'Low', 'Close']]

        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            with open('err.txt', 'a+') as f:
                print(ticker, e, file=f)

    return data


def save_to_csv(data_dict, folder):
    print(f'Saving {len(data_dict)} tickers in {folder}')
    os.makedirs(folder, exist_ok=True)
    for ticker, df in data_dict.items():
        print(f'Saving {ticker}, bars: {len(df)}, years: {len(df)/252:.1f}')
        df.to_csv(f"{folder}/{ticker}.csv")


if __name__ == "__main__":

    with open('err.txt', 'w') as f:
        pass

    # S&P 500
    tickers = get_sp500_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/sp500')
    save_to_csv(data, FOLDER + '/sp500')

    # Russell 1000
    # tickers = get_russell1000_tickers()
    # data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/russell1000')
    # save_to_csv(data, FOLDER + '/russell1000')

    # Dax 40
    tickers = get_dax40_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/dax40')
    save_to_csv(data, FOLDER + '/dax40')

    # Cac 40
    tickers = get_cac40_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/cac40')
    save_to_csv(data, FOLDER + '/cac40')

    # FTSE MIB
    tickers = get_ftsemib_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/ftsemib')
    save_to_csv(data, FOLDER + '/ftsemib')

    # FTSE Italia STAR
    # tickers = get_ftseItaliaSTAR_tickers()
    # data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/ftseItaliaSTAR')
    # save_to_csv(data, FOLDER + '/ftseItaliaSTAR')

    # FTSE 100
    tickers = get_ftse100_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/ftse100')
    save_to_csv(data, FOLDER + '/ftse100')

    # IBEX 35
    tickers = get_ibex35_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/ibex35')
    save_to_csv(data, FOLDER + '/ibex35')

    # Euronext 100
    tickers = get_euronext100_tickers()
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/euronext100')
    save_to_csv(data, FOLDER + '/euronext100')

    # Swiss SMI
    # tickers = get_smi_tickers()
    # data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/smi')
    # save_to_csv(data, FOLDER + '/smi')

    # Indices
    tickers = [ '^GSPC',   # S&P 500
                '^IXIC',   # NASDAQ Composite
                '^DJI',   # Dow Jones Industrial Average
                '^RUT',  # Russell 2000
                '^GDAXI',   # DAX
                '^FCHI',   # CAC 40
                '^FTSE',   # FTSE 100
                '^IBEX',   # Ibex 35
                'FTSEMIB.MI',  # FTSE MIB
                '^N100',   # Euronext 100 Index
                '^STOXX50E'   # EURO STOXX 50 I
               ]
    data = download_ohlcv_data(tickers, start=START_DATE, end=datetime.now().strftime('%Y-%m-%d'), check_folder=FOLDER + '/indices')
    save_to_csv(data, FOLDER + '/indices')

    if os.path.getsize('err.txt') == 0:
        os.remove('err.txt')
