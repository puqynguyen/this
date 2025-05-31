import sys
import os

# Thêm thư mục gốc của dự án (C:\Users\milaa\Desktop\Dev\Python\this) vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from torch.utils.data import Dataset
import torch
import pandas as pd

class StockDataset(Dataset):
    def __init__(self, data, seq_len, predict_day=None, mode='train'):
        self.data = data
        self.seq_len = seq_len
        self.predict_day = predict_day  # Số ngày từ ngày cuối cùng của sequence đến ngày dự đoán
        self.mode = mode  # 'train' hoặc 'predict'
        self.features = [col for col in data.columns if col not in ['date', 'last', 'pct_change_1d', 'pct_change_3d', 'pct_change_7d', 'movement_1d', 'movement_3d', 'movement_7d', 'day_of_week', 'day_of_month', 'month', 'year']]
        self.time_features = ['day_of_week', 'day_of_month', 'month', 'year']
        
        # Trong chế độ train, cần đủ dữ liệu để tính giá trị thực tế
        if self.mode == 'train':
            if self.predict_day is None:
                raise ValueError("predict_day phải được cung cấp trong chế độ train")
            self.valid_length = len(data) - seq_len - self.predict_day
        else:
            # Trong chế độ predict, chỉ cần đủ dữ liệu cho seq_len
            self.valid_length = len(data) - seq_len
        if self.valid_length <= 0:
            raise ValueError(f"Dữ liệu quá ngắn: {len(data)} hàng, cần ít nhất {seq_len + (self.predict_day if self.mode == 'train' else 0)}")
        
    def calculate_target(self, idx):
        if idx + self.seq_len + self.predict_day < len(self.data):
            future_price = self.data['last'].iloc[idx + self.seq_len + self.predict_day]
            if pd.isna(future_price):
                return 0.0  # Giá trị mặc định nếu không có dữ liệu
            return future_price
        else:
            return 0.0  # Giá trị mặc định nếu không đủ dữ liệu
    
    def __len__(self):
        return self.valid_length
    
    def __getitem__(self, idx):
        if idx >= self.valid_length:
            raise IndexError(f"Chỉ số {idx} vượt quá giới hạn của tập dữ liệu với độ dài {self.valid_length}")
        x = self.data[self.features].iloc[idx:idx+self.seq_len].values
        time_features = self.data[self.time_features].iloc[idx:idx+self.seq_len].values
        x_tensor = torch.tensor(x, dtype=torch.float32)
        time_tensor = torch.tensor(time_features, dtype=torch.float32)
        
        if self.mode == 'train':
            y = self.calculate_target(idx)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            # Kiểm tra NaN
            if torch.isnan(x_tensor).any() or torch.isnan(time_tensor).any() or torch.isnan(y_tensor).any():
                raise ValueError(f"Dữ liệu tại chỉ số {idx} chứa NaN")
            return x_tensor, time_tensor, y_tensor
        else:
            # Trong chế độ predict, không cần nhãn
            if torch.isnan(x_tensor).any() or torch.isnan(time_tensor).any():
                raise ValueError(f"Dữ liệu tại chỉ số {idx} chứa NaN")
            return x_tensor, time_tensor, torch.tensor(0.0, dtype=torch.float32)  # Placeholder cho chế độ predict