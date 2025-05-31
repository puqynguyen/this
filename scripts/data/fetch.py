import pymongo
from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION, MONGO_COLLECTION_PRICE
from datetime import datetime

def fetch_data(stock_symbol, start_date, end_date):
    # Kết nối MongoDB với timeout tăng
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=30000, socketTimeoutMS=30000)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    collection_price = db[MONGO_COLLECTION_PRICE]

    # Tạo index nếu chưa có
    collection.create_index([("stock_symbol", pymongo.ASCENDING), ("date", pymongo.ASCENDING)])
    collection_price.create_index([("stock_symbol", pymongo.ASCENDING), ("date", pymongo.ASCENDING)])

    # Lấy dữ liệu giá cổ phiếu
    price_query = {
        "stock_symbol": stock_symbol,
        "date": {"$gte": start_date.strftime('%Y-%m-%d'), "$lte": end_date.strftime('%Y-%m-%d')}
    }
    print(f"Truy vấn Price: {price_query}")
    price_cursor = collection_price.find(price_query).batch_size(1000)
    price_data = []
    for doc in price_cursor:
        price_data.append(doc)
    print(f"Số bản ghi Price tìm thấy: {len(price_data)}")

    # Lấy dữ liệu tin tức với projection
    news_query = {
        "stock_symbol": stock_symbol,
        "date": {"$gte": start_date.strftime('%Y-%m-%d'), "$lte": end_date.strftime('%Y-%m-%d')}
    }
    projection = {
        "stock_symbol": 1,
        "date": 1,
        "sentiment_score": 1,
        "impact_score": 1,
        "_id": 0
    }
    print(f"Truy vấn News: {news_query}")
    news_cursor = collection.find(news_query, projection).batch_size(1000)
    news_data = []
    count = 0
    for doc in news_cursor:
        news_data.append(doc)
        count += 1
        if count % 1000 == 0:
            print(f"Fetched {count} news records...")
    print(f"Số bản ghi News tìm thấy: {len(news_data)}")
    print("Xong nè")

    client.close()
    return price_data, news_data