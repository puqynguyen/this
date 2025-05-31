import sys
import os

# Thêm thư mục gốc của dự án (C:\Users\milaa\Desktop\Dev\Python\this) vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from scripts.config import DATA_DIR, MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE
import pandas as pd
from scripts.model.dataset import StockDataset
from scripts.model.transformer import StockTransformer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

def time_series_split(data, n_splits=5, val_days=8, seq_len=8, predict_day=1):
    """Chia dữ liệu theo Time Series Cross-Validation, chỉ tính các ngày giao dịch."""
    folds = []
    data = data.sort_values('date')
    dates = data['date'].unique()
    total_days = len(dates)

    # Kiểm tra tổng số ngày có đủ để chia fold không
    min_val_days = seq_len + predict_day + 1  # Cần thêm 1 ngày để tạo ít nhất 1 mẫu hợp lệ
    min_required_days = n_splits * min_val_days
    if total_days < min_required_days:
        print(f"Tổng số ngày ({total_days}) không đủ để chia {n_splits} fold với mỗi fold cần ít nhất {min_val_days} ngày.")
        n_splits = max(1, total_days // min_val_days)  # Điều chỉnh số fold
        print(f"Điều chỉnh số fold thành: {n_splits}")

    fold_size = total_days // (n_splits + 1)  # Chia đều dữ liệu

    for i in range(n_splits):
        train_end_idx = (i + 1) * fold_size
        val_end_idx = train_end_idx + val_days

        if val_end_idx >= total_days:
            val_end_idx = total_days

        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]

        train_fold = data[data['date'].isin(train_dates)]
        val_fold = data[data['date'].isin(val_dates)]

        # Kiểm tra số ngày trong tập validation
        if len(val_dates) < min_val_days:
            print(f"Fold {i+1}: Tập validation chỉ có {len(val_dates)} ngày, cần ít nhất {min_val_days} ngày. Bỏ qua fold này...")
            continue

        # Kiểm tra số ngày trong tập train
        if len(train_dates) < seq_len:
            print(f"Fold {i+1}: Tập train chỉ có {len(train_dates)} ngày, cần ít nhất {seq_len} ngày. Bỏ qua fold này...")
            continue

        folds.append((train_fold, val_fold))

    return folds

def train_model(stock_symbol, train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE, predict_day=1):
    # Kiểm tra dữ liệu huấn luyện và validation
    if train_data.empty:
        print(f"Không có dữ liệu huấn luyện cho {stock_symbol}")
        return None

    if val_data.empty:
        print(f"Không có dữ liệu validation cho {stock_symbol}")
        return None

    # Tính seq_len động dựa trên predict_day
    seq_len = predict_day * 2  # context_length = prediction_length * 2

    # Gộp train_data và val_data để sử dụng toàn bộ dữ liệu có tin tức
    data = pd.concat([train_data, val_data], ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])

    # Triển khai Time Series Cross-Validation
    n_splits = 5
    folds = time_series_split(data, n_splits=n_splits, val_days=8, seq_len=seq_len, predict_day=predict_day)
    print(f"Số fold trong Cross-Validation: {len(folds)}")

    if not folds:
        print("Không có fold nào hợp lệ để huấn luyện. Dừng lại...")
        return None

    model = StockTransformer(input_dim=len(StockDataset(data, seq_len, mode='predict').features))
    model_path = os.path.join(MODELS_DIR, f'model_{stock_symbol}.pt')
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Đã tải mô hình từ {model_path}")
    except FileNotFoundError:
        print(f"Không tìm thấy mô hình tại {model_path}, khởi tạo mô hình mới...")

    criterion = nn.MSELoss()  # Sử dụng MSE Loss cho dự đoán giá trị liên tục
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Thêm Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Thêm Early Stopping
    early_stopping_patience = 5
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Huấn luyện và xác thực trên từng fold
    window_size = 30  # Kích thước cửa sổ (30 ngày)
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/{len(folds)}")

        try:
            train_dataset = StockDataset(train_fold, seq_len=seq_len, predict_day=predict_day, mode='train')
            val_dataset = StockDataset(val_fold, seq_len=seq_len, predict_day=predict_day, mode='train')
        except ValueError as e:
            print(f"Lỗi khi tạo dataset cho fold {fold_idx + 1}: {str(e)}")
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Rolling Window trên tập train_fold
        train_dates = train_fold['date'].unique()
        total_train_days = len(train_dates)

        for window_start in range(0, total_train_days - window_size + 1, 1):  # Step = 1 để di chuyển từng ngày
            window_end = window_start + window_size
            if window_end > total_train_days:
                window_end = total_train_days

            window_dates = train_dates[window_start:window_end]
            window_train_data = train_fold[train_fold['date'].isin(window_dates)]

            try:
                window_train_dataset = StockDataset(window_train_data, seq_len=seq_len, predict_day=predict_day, mode='train')
                window_train_loader = DataLoader(window_train_dataset, batch_size=batch_size, shuffle=False)
            except ValueError as e:
                print(f"Lỗi khi tạo dataset huấn luyện cho window {window_start}-{window_end}: {str(e)}")
                continue

            # Huấn luyện mô hình trên cửa sổ hiện tại
            model.train()
            for batch_idx, (x, time_features, y) in enumerate(window_train_loader):
                if torch.isnan(x).any() or torch.isnan(time_features).any() or torch.isnan(y).any():
                    print(f"Phát hiện NaN trong batch tại window {window_start}-{window_end}, bỏ qua batch này...")
                    continue
                optimizer.zero_grad()
                output = model(x, time_features)  # Shape: (batch_size,)
                if torch.isnan(output).any():
                    print(f"Output chứa NaN tại window {window_start}-{window_end}, bỏ qua batch này...")
                    continue

                loss = criterion(output, y)
                if torch.isnan(loss):
                    print(f"Loss là NaN tại window {window_start}-{window_end}, bỏ qua batch này...")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Giới hạn gradient
                optimizer.step()

        # Đánh giá trên tập validation của fold hiện tại
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (x, time_features, y) in enumerate(val_loader):
                if torch.isnan(x).any() or torch.isnan(time_features).any() or torch.isnan(y).any():
                    print(f"Phát hiện NaN trong batch validation tại fold {fold_idx + 1}, bỏ qua batch này...")
                    continue
                output = model(x, time_features)  # Shape: (batch_size,)
                if torch.isnan(output).any():
                    print(f"Output chứa NaN trong batch validation tại fold {fold_idx + 1}, bỏ qua batch này...")
                    continue

                loss = criterion(output, y)
                val_loss += loss.item()
                val_batch_count += 1

                all_preds.extend(output.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        if val_batch_count == 0:
            print(f"Không có batch hợp lệ nào trong tập validation tại fold {fold_idx + 1}, bỏ qua...")
            continue

        val_loss /= val_batch_count
        print(f"Fold {fold_idx + 1}, Validation Loss: {val_loss}")

        # Tính toán các chỉ số đánh giá
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        print(f"Fold {fold_idx + 1}, Validation MAE: {mae:.4f}")
        print(f"Fold {fold_idx + 1}, Validation RMSE: {rmse:.4f}")

        # Điều chỉnh LR
        scheduler.step()

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Đã lưu mô hình tốt nhất tại {model_path} với Validation Loss: {best_val_loss}")
        else:
            early_stopping_counter += 1
            print(f"Early Stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print("Early Stopping triggered. Stopping training...")
                break

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dự đoán giá cổ phiếu")
    parser.add_argument('--stock', type=str, required=True, help="Mã cổ phiếu (ví dụ: AAPL)")
    args = parser.parse_args()

    stock_symbol = args.stock
    model = train_model(stock_symbol)
    if model:
        print(f"Hoàn tất huấn luyện mô hình cho {stock_symbol}.")