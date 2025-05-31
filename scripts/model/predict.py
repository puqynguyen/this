import sys
import os

# Thêm thư mục gốc của dự án (C:\Users\milaa\Desktop\Dev\Python\this) vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch
import pandas as pd
import numpy as np
import json
import joblib
import pymongo
from scripts.config import DATA_DIR, MODELS_DIR, MONGO_URI, MONGO_DB, MONGO_COLLECTION_PRICE
from scripts.model.dataset import StockDataset
from scripts.model.transformer import StockTransformer
from datetime import datetime, timedelta

def fetch_actual_price(stock_symbol, date):
    """Truy xuất giá thực tế từ MongoDB cho ngày cụ thể."""
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection_price = db[MONGO_COLLECTION_PRICE]

    date = pd.to_datetime(date)
    query = {
        "stock_symbol": stock_symbol,
        "date": date.strftime('%Y-%m-%d')
    }
    price_data = collection_price.find_one(query)

    client.close()

    if price_data and 'last' in price_data:
        return float(price_data['last'])
    return None

def predict(stock_symbol, date):
    # Đọc dữ liệu huấn luyện
    train_file = os.path.join(DATA_DIR, f'processed_data_{stock_symbol}_train.csv')
    
    # Kiểm tra file dữ liệu
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu cho {stock_symbol}: {train_file}")

    # Đọc dữ liệu
    data = pd.read_csv(train_file)
    data['date'] = pd.to_datetime(data['date'])

    # Kiểm tra ngày nhập
    input_date = pd.to_datetime(date)
    min_date = data['date'].min()
    if input_date < min_date:
        raise ValueError(f"Ngày {date} nhỏ hơn ngày nhỏ nhất trong dữ liệu ({min_date.strftime('%Y-%m-%d')})")

    # Tính predict_day dựa trên ngày nhập
    last_sequence_date = data['date'].max()
    predict_day = (input_date - last_sequence_date).days
    if predict_day <= 0:
        raise ValueError(f"Ngày dự đoán {date} phải sau ngày cuối cùng trong dữ liệu ({last_sequence_date.strftime('%Y-%m-%d')})")

    # Tính seq_len động dựa trên predict_day
    seq_len = predict_day * 2  # context_length = prediction_length * 2

    # Tải scaler để đảo ngược quá trình scale
    scaler_path = os.path.join(DATA_DIR, f'scaler_price_{stock_symbol}.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Không tìm thấy file scaler: {scaler_path}")
    scaler_price = joblib.load(scaler_path)

    # Tải mô hình đã huấn luyện
    model_path = os.path.join(MODELS_DIR, f'model_{stock_symbol}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình cho {stock_symbol} tại {model_path}")

    model = StockTransformer(input_dim=len(StockDataset(data, seq_len, mode='predict').features))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Chuẩn bị dữ liệu để dự đoán
    dataset = StockDataset(data, seq_len, mode='predict')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Tìm vị trí của ngày nhập trong dữ liệu
    last_sequence_date = input_date - timedelta(days=seq_len)
    date_idx = data[data['date'] <= last_sequence_date].index[-1] - (seq_len - 1)
    if date_idx < 0:
        raise ValueError(f"Không đủ dữ liệu lịch sử (cần ít nhất {seq_len} ngày trước ngày {last_sequence_date.strftime('%Y-%m-%d')}) để dự đoán")

    # Lấy dữ liệu để dự đoán
    batch_idx = date_idx // 32
    within_batch_idx = date_idx % 32

    predictions = {}
    actual = {}
    with torch.no_grad():
        for i, (x, time_features, _) in enumerate(dataloader):
            if i != batch_idx:
                continue
            output = model(x, time_features)  # Shape: (batch_size,)
            pred_value = output[within_batch_idx].item()

            # Đảo ngược quá trình scale cho giá dự báo
            scaled_price_array = np.array([[pred_value, 0, 0, 0, 0, 0]])  # Cần đúng shape của scaler (6 cột: last, open, high, low, volume, %change)
            unscaled_price_array = scaler_price.inverse_transform(scaled_price_array)
            unscaled_price = unscaled_price_array[0][0]  # Lấy giá last đã đảo ngược

            predictions = {
                "date": input_date.strftime('%Y-%m-%d'),
                "price": float(unscaled_price)
            }

            # Truy xuất giá thực tế từ MongoDB
            actual_price = fetch_actual_price(stock_symbol, input_date)
            if actual_price is not None:
                actual = {
                    "price": actual_price
                }
            else:
                actual = {
                    "price": None
                }
            break

    # Trả về kết quả dự đoán
    result = {
        "date": input_date.strftime('%Y-%m-%d'),
        "predictions": predictions,
        "actual": actual
    }
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dự đoán giá cổ phiếu")
    parser.add_argument('--stock', type=str, required=True, help="Mã cổ phiếu (ví dụ: AAPL)")
    parser.add_argument('--date', type=str, required=True, help="Ngày cần dự đoán (YYYY-MM-DD)")
    args = parser.parse_args()

    try:
        result = predict(args.stock, args.date)
        print(json.dumps(result, indent=4))
    except Exception as e:
        error = {"error": str(e)}
        print(json.dumps(error, indent=4))