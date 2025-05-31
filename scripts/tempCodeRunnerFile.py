import pymongo
from data import fetch_data, process_data
from model import train_model, simulate
from datetime import datetime, timedelta
from config import DATA_DIR, MONGO_URI, MONGO_DB, MONGO_COLLECTION_PRICE

def get_all_stock_symbols():
    # Kết nối MongoDB để lấy danh sách mã cổ phiếu
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection_price = db[MONGO_COLLECTION_PRICE]

    # Lấy danh sách tất cả mã cổ phiếu
    stock_symbols = collection_price.distinct("stock_symbol")
    print(f"Danh sách mã cổ phiếu: {stock_symbols}")

    client.close()
    return stock_symbols

def main():
    # Lấy danh sách tất cả mã cổ phiếu
    stock_symbols = get_all_stock_symbols()

    # Lặp qua từng mã cổ phiếu
    for stock_symbol in stock_symbols:
        print(f"\n=== Bắt đầu xử lý mã cổ phiếu: {stock_symbol} ===")

        # Bước 1: Lấy dữ liệu để xác định khoảng thời gian
        print(f"Đang lấy dữ liệu cho mã cổ phiếu: {stock_symbol}")
        price_data_all, _ = fetch_data(stock_symbol, datetime(1970, 1, 1), datetime(2025, 12, 31))
        print("Xong nè")
        if not price_data_all:
            print(f"Không tìm thấy dữ liệu giá cho {stock_symbol} trong MongoDB, bỏ qua...")
            continue

        # Xác định ngày đầu và ngày cuối từ dữ liệu
        price_data_all = sorted(price_data_all, key=lambda x: x['date'])
        first_record = price_data_all[0]
        last_record = price_data_all[-1]
        start_date = datetime.strptime(first_record['date'], '%Y-%m-%d')
        end_date = datetime.strptime(last_record['date'], '%Y-%m-%d')
        split_date = end_date - timedelta(days=67)

        # Bước 2: Lấy và xử lý dữ liệu
        print(f"Đang xử lý dữ liệu cho mã cổ phiếu: {stock_symbol}")
        price_data, news_data = fetch_data(stock_symbol, start_date, end_date)
        print(f"Số bản ghi giá: {len(price_data)}, Số bản ghi tin tức: {len(news_data)}")
        train_data, sim_data = process_data(stock_symbol, price_data, news_data, start_date, end_date, split_date)
        if train_data is None or sim_data is None:
            print(f"Không có dữ liệu để xử lý cho {stock_symbol}, bỏ qua...")
            continue

        print(f"Đã lưu dữ liệu huấn luyện vào {DATA_DIR}/processed_data_{stock_symbol}_train.csv")
        print(f"Đã lưu dữ liệu mô phỏng vào {DATA_DIR}/processed_data_{stock_symbol}_sim.csv")

        # Bước 3: Huấn luyện mô hình
        print(f"Đang huấn luyện mô hình cho {stock_symbol}...")
        model = train_model(stock_symbol)
        if model is None:
            print(f"Không thể huấn luyện mô hình cho {stock_symbol}, bỏ qua...")
            continue

        # Bước 4: Mô phỏng
        print(f"Đang mô phỏng cho {stock_symbol}...")
        simulate(stock_symbol)

        print(f"=== Hoàn tất xử lý mã cổ phiếu: {stock_symbol} ===\n")

    print("Hoàn tất xử lý cho tất cả mã cổ phiếu!")

if __name__ == "__main__":
    main()