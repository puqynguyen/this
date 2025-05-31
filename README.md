C:\Users\milaa\Desktop\Dev\Python\this\
  ├── data/                           # Thư mục lưu trữ dữ liệu đã xử lý
  │   ├── processed_data_AAPL_train.csv   # Dữ liệu huấn luyện cho AAPL
  │   ├── processed_data_AAPL_sim.csv     # Dữ liệu mô phỏng cho AAPL
  │   ├── predictions_AAPL.json           # Kết quả dự đoán cho AAPL
  │   ├── scaler_price_AAPL.pkl           # File chuẩn hóa giá cho AAPL
  │   ├── scaler_sentiment_AAPL.pkl       # File chuẩn hóa sentiment cho AAPL
  │   └── ...                             # Tương tự cho các mã cổ phiếu khác (BID, FPT, META, NVDA, VCB)
  ├── models/                         # Thư mục lưu trữ mô hình đã huấn luyện
  │   ├── model_AAPL.pt               # Mô hình đã huấn luyện cho AAPL
  │   └── ...                         # Tương tự cho các mã cổ phiếu khác (BID, FPT, META, NVDA, VCB)
  ├── scripts/                        # Thư mục chứa các script Python
  │   ├── main.py                 # File chính để chạy pipeline
  │   ├── data/                   # Module xử lý dữ liệu
  │   │   ├── __init__.py
  │   │   ├── fetch.py            # Lấy dữ liệu từ MongoDB
  │   │   ├── process.py          # Xử lý dữ liệu (điền ngày thiếu, chuẩn hóa, v.v.)
  │   │   └── utils.py            # Các hàm tiện ích (chuẩn hóa, tính biến động)
  │   ├── model/                  # Module huấn luyện và dự đoán mô hình
  │   │   ├── __init__.py
  │   │   ├── dataset.py          # Định nghĩa StockDataset
  │   │   ├── transformer.py      # Định nghĩa StockTransformer
  │   │   ├── train.py            # Logic huấn luyện mô hình
  │   │   ├── predict.py          # Logic dự đoán biến động giá
  │   │   └── simulate.py         # Logic mô phỏng
  │   ├── db/                     # Module xử lý dữ liệu MongoDB
  │   │   ├── __init__.py
  │   │   └── connect.py          # Kết nối MongoDB
  │   └── config.py               # File chứa cấu hình (đường dẫn, tham số, v.v.)
  ├── web/                            # Thư mục chứa code phía server và client
  │   ├── server.js                   # File chính khởi tạo Express và routes
  │   ├── routes/                     # Thư mục chứa các routes (API endpoints)
  │   │   ├── stocks.js               # Route cho /api/stocks
  │   │   ├── stockData.js            # Route cho /api/stock/:symbol/:range
  │   │   └── simulate.js             # Route cho /api/simulate/:symbol
  │   ├── utils/                      # Thư mục chứa các hàm tiện ích
  │   │   └── db.js                   # Logic kết nối MongoDB
  │   └── public/                     # Thư mục chứa các file tĩnh (client-side)
  │       ├── index.html              # File HTML chính
  │       ├── css/                    # Thư mục chứa CSS
  │       │   └── styles.css          # CSS styles
  │       └── js/                     # Thư mục chứa JavaScript
  │           ├── chartConfig.js      # Cấu hình biểu đồ (Chart.js)
  │           ├── api.js              # Hàm gọi API
  │           ├── simulation.js       # Logic mô phỏng
  │           └── main.js             # Logic chính (khởi tạo trang, xử lý sự kiện)
  ├── .env                            # File chứa biến môi trường (MONGO_URI, v.v.)
  └── README.md                       # File mô tả dự án

# Dự án Mô phỏng Biến động Giá Cổ Phiếu

## Cấu trúc thư mục
- `data/`: Lưu trữ dữ liệu đã xử lý và kết quả dự đoán.
- `models/`: Lưu trữ mô hình đã huấn luyện.
- `scripts/`: Chứa các script Python.
- `web/`: Chứa code phía server và client.

## Cách chạy
1. Cài đặt các thư viện cần thiết:
pip install pymongo pandas sklearn torch npm install express mongodb dotenv
2. Chạy script chính:
python scripts/main.py
3. Khởi động server:
cd web node server.js
4. Truy cập http://localhost:3000 để xem giao diện.