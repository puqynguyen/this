import pandas as pd
from datetime import datetime, timedelta
import os
from config import DATA_DIR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import numpy as np
from .utils import fill_missing_dates

def adjust_to_friday(date):
    """Chuyển ngày cuối tuần (thứ 7, chủ nhật) về ngày thứ 6 của tuần đó."""
    date = pd.to_datetime(date)
    weekday = date.weekday()  # 0: Thứ 2, 6: Chủ nhật
    if weekday == 5:  # Thứ 7
        return date - timedelta(days=1)
    elif weekday == 6:  # Chủ nhật
        return date - timedelta(days=2)
    return date

def process_data(stock_symbol, price_data, news_data, start_date, end_date, split_date):
    # Xử lý dữ liệu giá
    price_df = pd.DataFrame(price_data)
    if price_df.empty:
        return None, None

    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date')

    # Lọc bỏ các ngày không giao dịch (thứ 7, chủ nhật)
    price_df['weekday'] = price_df['date'].dt.weekday
    price_df = price_df[price_df['weekday'] < 5]  # Chỉ giữ thứ 2 đến thứ 6
    price_df = price_df.drop(columns=['weekday'])

    # Điền ngày thiếu (chỉ cho các ngày từ thứ 2 đến thứ 6)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' là Business day (thứ 2 - thứ 6)
    price_df_full = pd.DataFrame({'date': date_range})
    price_df_full = price_df_full.merge(price_df, on='date', how='left')

    # Thêm cột is_trading_day và kiểm tra NaN
    price_df_full['is_trading_day'] = price_df_full['_id'].notnull().astype(int)
    price_cols = ['last', 'open', 'high', 'low', 'volume', '%change']
    for col in price_cols:
        if col not in price_df_full.columns:
            price_df_full[col] = np.nan

    # Debug: Tìm vị trí NaN trong dữ liệu giá
    nan_rows = price_df_full[price_df_full[price_cols].isna().any(axis=1)]
    if not nan_rows.empty:
        print(f"Các ngày chứa NaN trong dữ liệu giá cho {stock_symbol}:")
        print(nan_rows[['date'] + price_cols])

    # Lọc bỏ các hàng chứa NaN trong các cột giá
    price_df_full = price_df_full.dropna(subset=price_cols)

    # Xử lý dữ liệu tin tức
    news_df = pd.DataFrame(news_data)
    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['date'])

        # Chuyển các ngày cuối tuần về thứ 6
        news_df['date'] = news_df['date'].apply(adjust_to_friday)

        # Chuyển đổi các cột boolean thành số (0/1)
        bool_cols = ['direct_impact', 'target_mentioned']
        for col in bool_cols:
            if col in news_df.columns:
                news_df[col] = news_df[col].astype(int)

        # Chuyển đổi cột sentiment_label thành số
        if 'sentiment_label' in news_df.columns:
            sentiment_mapping = {'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': -1}
            news_df['sentiment_label'] = news_df['sentiment_label'].map(sentiment_mapping).fillna(0)

        # Gom nhóm theo ngày và lấy trung bình các giá trị số (bao gồm sentiment)
        news_daily = news_df.groupby(news_df['date'].dt.date).mean(numeric_only=True).reset_index()
        news_daily['date'] = pd.to_datetime(news_daily['date'])
        news_df_full = price_df_full[['date']].merge(news_daily, on='date', how='left')
        news_df_full['has_news'] = news_df_full['sentiment_score'].notnull().astype(int)

        # Xác định khoảng thời gian có tin tức
        news_dates = news_df_full[news_df_full['has_news'] == 1]['date']
        if not news_dates.empty:
            min_news_date = news_dates.min()
            max_news_date = news_dates.max()
            print(f"Khoảng thời gian có tin tức cho {stock_symbol}: từ {min_news_date} đến {max_news_date}")

            # Chỉ giữ các ngày trong khoảng thời gian có tin tức
            news_period_mask = (news_df_full['date'] >= min_news_date) & (news_df_full['date'] <= max_news_date)
            news_df_full = news_df_full[news_period_mask]
            price_df_full = price_df_full[price_df_full['date'].isin(news_df_full['date'])]

            # Tính ngày currentDay (ngày trước 60 ngày cuối cùng trong khoảng thời gian có tin tức)
            max_news_date = pd.to_datetime(max_news_date)
            new_end_date = max_news_date
            new_split_date = new_end_date - timedelta(days=60)  # Current day
            print(f"Current day (ngày trước 60 ngày cuối): {new_split_date}")
        else:
            print(f"Không có dữ liệu tin tức cho {stock_symbol}, bỏ qua...")
            return None, None
    else:
        print(f"Không có dữ liệu tin tức cho {stock_symbol}, bỏ qua...")
        return None, None

    # Điền giá trị mặc định cho các cột tin tức
    sentiment_cols = [
        'medium_term_score', 'prominence', 'topic_regulatory_news', 'topic_ai_sector',
        'topic_corporate_changes', 'sentiment_score', 'topic_financial_performance',
        'topic_investor_sentiment', 'topic_competition', 'impact_score',
        'topic_market_conditions', 'timeliness', 'confidence', 'topic_technology',
        'short_term_score', 'relevance_score', 'direct_impact', 'target_mentioned',
        'sentiment_label', 'has_news'
    ]
    for col in sentiment_cols:
        if col not in news_df_full.columns:
            news_df_full[col] = 0
        news_df_full[col] = news_df_full[col].fillna(0)

    # Gộp dữ liệu
    data = price_df_full.merge(news_df_full, on='date')
    data = data.drop(columns=['_id', 'stock_symbol'], errors='ignore')

    # Thêm các đặc trưng thời gian
    data['day_of_week'] = data['date'].dt.weekday
    data['day_of_month'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Kiểm tra NaN sau khi gộp
    if data.isna().any().any():
        print(f"Cảnh báo: Dữ liệu cho {stock_symbol} vẫn chứa NaN sau khi gộp!")
        print(data.isna().sum())

    # Chuẩn hóa dữ liệu giá trên toàn bộ dữ liệu
    scaler_price = StandardScaler()  # Chuyển sang StandardScaler theo Autoformer
    price_cols = ['last', 'open', 'high', 'low', 'volume', '%change']
    data[price_cols] = scaler_price.fit_transform(data[price_cols])
    joblib.dump(scaler_price, os.path.join(DATA_DIR, f'scaler_price_{stock_symbol}.pkl'))

    # Chuẩn hóa dữ liệu tin tức trên toàn bộ dữ liệu
    scaler_sentiment = MinMaxScaler()
    data[sentiment_cols] = scaler_sentiment.fit_transform(data[sentiment_cols])
    joblib.dump(scaler_sentiment, os.path.join(DATA_DIR, f'scaler_sentiment_{stock_symbol}.pkl'))

    # Chuẩn hóa các đặc trưng thời gian
    time_features = ['day_of_week', 'day_of_month', 'month', 'year']
    scaler_time = MinMaxScaler()
    data[time_features] = scaler_time.fit_transform(data[time_features])
    joblib.dump(scaler_time, os.path.join(DATA_DIR, f'scaler_time_{stock_symbol}.pkl'))

    # Chia dữ liệu thành tập train và val (80% train, 20% val)
    total_days = len(data)
    train_size = int(0.8 * total_days)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    # Lưu dữ liệu đã xử lý
    train_output_file = os.path.join(DATA_DIR, f'processed_data_{stock_symbol}_train.csv')
    val_output_file = os.path.join(DATA_DIR, f'processed_data_{stock_symbol}_val.csv')
    train_data.to_csv(train_output_file, index=False)
    val_data.to_csv(val_output_file, index=False)

    print(f"Đã lưu dữ liệu huấn luyện vào {train_output_file}")
    print(f"Đã lưu dữ liệu validation vào {val_output_file}")

    return train_data, val_data