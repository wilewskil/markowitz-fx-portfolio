import MetaTrader5 as mt5 # type:  ignore
import pandas as pd
import numpy as np
import datetime as dt

def connect_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    else:
        print("MT5 initialized successfully")
        return True

def get_fx_data(symbol, timeframe, start_date, end_date):
    if not connect_mt5():
        return None

    # Convert dates to datetime objects
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")

    # Get data from MT5
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    if rates is None:
        print("No data retrieved, error code =", mt5.last_error())
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    mt5.shutdown()
    return df


def get_all_pairs(symbols, timeframe, start_date, end_date):
    all_close = {}
    for symbol in symbols:
        df = get_fx_data(symbol, timeframe, start_date, end_date)
        if df is not None:
            all_close[symbol] = df['close']
        else:
            print(f"Failed to retrieve data for {symbol}")
    return pd.DataFrame(all_close)


def calculate_log_returns(df):
    log_returns = np.log(df / df.shift(1))
    return log_returns


if __name__ == "__main__":
    symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'USDJPY']
    timeframe = mt5.TIMEFRAME_H4
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    df = get_all_pairs(symbols, timeframe, start_date, end_date)

    log_returns = calculate_log_returns(df)
    print(log_returns.head())
    print(log_returns.describe())


    print(df.head())
    print(df)
    print(df.shape)



