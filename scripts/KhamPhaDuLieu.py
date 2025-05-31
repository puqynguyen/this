import os
from dotenv import load_dotenv
import pymongo
import pandas as pd
from tabulate import tabulate

# Tải biến môi trường từ file .env
load_dotenv()

# Kết nối MongoDB
mongo_uri = os.getenv("MONGO_URI")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")
mongo_collection_price = os.getenv("MONGO_COLLECTION_PRICE")

client = pymongo.MongoClient(mongo_uri)
db = client[mongo_db]
collection = db[mongo_collection]
collection_price = db[mongo_collection_price]

# Hàm chuyển đổi KL (volume) từ string sang số
def convert_volume(vol):
    if isinstance(vol, str):
        if 'M' in vol:
            return float(vol.replace('M', '')) * 1_000_000
        elif 'K' in vol:
            return float(vol.replace('K', '')) * 1_000
        else:
            return float(vol)
    return vol

# Hàm chuyển đổi giá trị từ chuỗi sang số (loại bỏ dấu phẩy nếu có)
def convert_to_float(value):
    if isinstance(value, str):
        # Loại bỏ dấu phẩy và chuyển thành số
        return float(value.replace(',', ''))
    return float(value)

# Lấy danh sách tất cả mã cổ phiếu
stock_symbols = collection_price.distinct("stock_symbol")

# Khám phá dữ liệu giá cổ phiếu
price_dfs = {}
for symbol in stock_symbols:
    price_data = list(collection_price.find({"stock_symbol": symbol}))
    if price_data:
        df = pd.DataFrame(price_data)
        df['date'] = pd.to_datetime(df['date'])
        # Chuyển đổi các cột last, open, high, low từ chuỗi sang số
        for col in ['last', 'open', 'high', 'low']:
            df[col] = df[col].apply(convert_to_float)
        df['KL'] = df['KL'].apply(convert_volume)
        df['%change'] = df['%change'].replace('%', '', regex=True).astype(float)
        df = df.sort_values('date')
        price_dfs[symbol] = df

# Khám phá dữ liệu tin tức sentiment
news_dfs = {}
for symbol in stock_symbols:
    news_data = list(collection.find({"stock_symbol": symbol}))
    if news_data:
        df = pd.DataFrame(news_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        news_dfs[symbol] = df

# Hàm hiển thị kết quả
def print_eda_results():
    # Tổng quan dữ liệu giá cổ phiếu
    print("=== TỔNG QUAN DỮ LIỆU GIÁ CỔ PHIẾU ===")
    if not price_dfs:
        print("Không có dữ liệu giá cổ phiếu.")
    else:
        print(f"Số lượng mã cổ phiếu: {len(price_dfs)}")
        all_dates = pd.concat([df['date'] for df in price_dfs.values()])
        print(f"Khoảng thời gian tổng: {all_dates.min()} đến {all_dates.max()}")
        
        # Thống kê theo từng mã cổ phiếu
        summary_data = []
        for symbol, df in price_dfs.items():
            summary = {
                "Mã": symbol,
                "Số bản ghi": len(df),
                "Bắt đầu": df['date'].min(),
                "Kết thúc": df['date'].max(),
                "Giá trung bình": df['last'].mean(),
                "KL trung bình": df['KL'].mean(),
                "%change trung bình": df['%change'].mean(),
                "Ngày không liên tục": (df['date'].diff().dt.days > 1).sum()
            }
            summary_data.append(summary)
        
        print("\nThống kê giá cổ phiếu theo mã:")
        print(tabulate(summary_data, headers="keys", tablefmt="psql", floatfmt=".2f"))

        # Dữ liệu thiếu
        print("\nDữ liệu thiếu (tổng cộng):")
        missing_data = []
        for symbol, df in price_dfs.items():
            missing = df[['last', 'open', 'high', 'low', 'KL', '%change']].isnull().sum()
            missing_data.append({"Mã": symbol, **missing})
        print(tabulate(missing_data, headers="keys", tablefmt="psql"))

    # Tổng quan dữ liệu tin tức sentiment
    print("\n=== TỔNG QUAN DỮ LIỆU TIN TỨC SENTIMENT ===")
    if not news_dfs:
        print("Không có dữ liệu tin tức.")
    else:
        print(f"Số lượng mã cổ phiếu có tin tức: {len(news_dfs)}")
        all_dates = pd.concat([df['date'] for df in news_dfs.values()])
        print(f"Khoảng thời gian tổng: {all_dates.min()} đến {all_dates.max()}")

        # Thống kê theo từng mã cổ phiếu
        summary_data = []
        sentiment_cols = ['medium_term_score', 'sentiment_score', 'relevance_score', 
                         'impact_score', 'topic_market_conditions', 'confidence']
        for symbol, df in news_dfs.items():
            daily_df = df.groupby(df['date'].dt.date).mean(numeric_only=True).reset_index()
            summary = {
                "Mã": symbol,
                "Số bản ghi": len(df),
                "Số ngày có tin tức": len(daily_df),
                "Sentiment trung bình": daily_df['sentiment_score'].mean(),
                "Relevance trung bình": daily_df['relevance_score'].mean(),
                "Impact trung bình": daily_df['impact_score'].mean()
            }
            summary_data.append(summary)
        
        print("\nThống kê tin tức theo mã:")
        print(tabulate(summary_data, headers="keys", tablefmt="psql", floatfmt=".2f"))

        # Phân bố sentiment_label
        print("\nPhân bố sentiment_label (tổng cộng):")
        all_labels = pd.concat([df['sentiment_label'] for df in news_dfs.values()])
        label_counts = all_labels.value_counts().reset_index()
        print(tabulate(label_counts, headers=['Label', 'Số lượng'], tablefmt="psql"))

    # Tương quan giữa giá và sentiment
    print("\n=== TƯƠNG QUAN GIÁ VÀ SENTIMENT ===")
    correlation_data = []
    for symbol in stock_symbols:
        if symbol in price_dfs and symbol in news_dfs:
            price_df = price_dfs[symbol][['date', 'last', '%change']]
            news_df = news_dfs[symbol]
            news_daily = news_df.groupby(news_df['date'].dt.date)[sentiment_cols].mean().reset_index()
            news_daily['date'] = pd.to_datetime(news_daily['date'])
            
            merged_df = pd.merge(price_df, news_daily, on='date', how='inner')
            if not merged_df.empty:
                corr = merged_df[['last', '%change'] + sentiment_cols].corr()['last'][sentiment_cols].to_dict()
                correlation_data.append({"Mã": symbol, **corr})

    if correlation_data:
        print("\nTương quan giữa giá đóng cửa (last) và sentiment:")
        print(tabulate(correlation_data, headers="keys", tablefmt="psql", floatfmt=".2f"))
    else:
        print("Không có dữ liệu để tính tương quan.")

# Thực thi EDA
print_eda_results()

# Đóng kết nối MongoDB
client.close()