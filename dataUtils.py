import pandas as pd
import pandas_datareader.data as web
import datetime as dt

def save_to_csv_from_yahoo(ticker, syear, smonth, sday, eyear, emonth, eday):
    start = dt.datetime(syear, smonth, sday)
    end = dt.datetime(eyear, emonth, eday)
    
    df = web.DataReader(ticker, 'yahoo', start, end)
    
    df.to_csv("/Users/PCComp/Documents/TFG/data/" + ticker + '.csv')
    return df

def get_df_from_csv(ticker):
    try:
        df = pd.read_csv("/Users/PCComp/Documents/TFG/data/" + ticker + '.csv', index_col=6)
    except FileNotFoundError:
        print("File Not Found")
    else:
        return df