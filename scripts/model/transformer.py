import torch
import torch.nn as nn
import math

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=3):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2  # Đảm bảo output có cùng độ dài với input
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # Tách trend bằng trung bình động
        trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # Shape: (batch_size, seq_len, d_model)
        seasonal = x - trend  # Seasonal là phần còn lại sau khi trừ trend
        return trend, seasonal

class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(EnhancedPositionalEncoding, self).__init__()
        # Positional Encoding truyền thống
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

        # Linear layer để kết hợp time features
        self.time_feature_dim = 4  # day_of_week, day_of_month, month, year
        self.time_feature_fc = nn.Linear(self.time_feature_dim, d_model)

    def forward(self, x, time_features):
        # x: (batch_size, seq_len, d_model)
        # time_features: (batch_size, seq_len, time_feature_dim)
        seq_len = x.size(1)
        # Thêm positional encoding truyền thống
        pe = self.pe[:, :seq_len, :].to(x.device)
        
        # Kết hợp time features
        time_features = self.time_feature_fc(time_features)  # Shape: (batch_size, seq_len, d_model)
        x = x + pe + time_features
        return x

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        energy = self.tanh(self.W(x))  # Shape: (batch_size, seq_len, d_model)
        attention_weights = torch.softmax(self.V(energy), dim=1)  # Shape: (batch_size, seq_len, 1)
        output = x * attention_weights  # Shape: (batch_size, seq_len, d_model)
        return output

class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.decomposition = SeriesDecomposition(kernel_size=3)  # Tầng phân tách trend và seasonal
        self.positional_encoding = EnhancedPositionalEncoding(d_model)
        self.temporal_attention = TemporalAttention(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.trend_fc = nn.Linear(d_model, d_model)  # Xử lý trend
        self.seasonal_fc = nn.Linear(d_model, d_model)  # Xử lý seasonal
        self.output_fc = nn.Linear(d_model * 2, 1)  # Kết hợp trend và seasonal để dự đoán giá trị liên tục
        
    def forward(self, src, time_features):
        # src: (batch_size, seq_len, input_dim)
        # time_features: (batch_size, seq_len, time_feature_dim)
        src = self.input_fc(src)  # Shape: (batch_size, seq_len, d_model)
        
        # Phân tách trend và seasonal
        trend, seasonal = self.decomposition(src)  # Shape: (batch_size, seq_len, d_model)
        
        # Thêm Positional Encoding với time features
        trend = self.positional_encoding(trend, time_features)
        seasonal = self.positional_encoding(seasonal, time_features)
        
        # Áp dụng Temporal Attention
        trend = self.temporal_attention(trend)
        seasonal = self.temporal_attention(seasonal)
        
        trend = self.dropout(trend)
        seasonal = self.dropout(seasonal)
        
        # Transformer
        trend = self.transformer(trend, trend)  # Shape: (batch_size, seq_len, d_model)
        seasonal = self.transformer(seasonal, seasonal)  # Shape: (batch_size, seq_len, d_model)
        
        # Lấy vector cuối cùng
        trend = trend[:, -1, :]  # Shape: (batch_size, d_model)
        seasonal = seasonal[:, -1, :]  # Shape: (batch_size, d_model)
        
        # Xử lý trend và seasonal riêng biệt
        trend = self.trend_fc(trend)  # Shape: (batch_size, d_model)
        seasonal = self.seasonal_fc(seasonal)  # Shape: (batch_size, d_model)
        
        # Kết hợp trend và seasonal
        combined = torch.cat([trend, seasonal], dim=-1)  # Shape: (batch_size, d_model * 2)
        output = self.output_fc(combined)  # Shape: (batch_size, 1)
        return output.squeeze(-1)  # Shape: (batch_size,)