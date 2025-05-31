import pandas as pd
import numpy as np

def fill_missing_dates(price_df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    price_df_full = pd.DataFrame({'date': date_range})
    price_df_full = price_df_full.merge(price_df, on='date', how='left')
    return price_df_full
