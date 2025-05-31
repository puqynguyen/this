import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env ở thư mục gốc
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Đường dẫn thư mục
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Tham số cấu hình
PREDICT_DAYS = [1, 3, 7]  # Các khoảng thời gian dự đoán
EPOCHS = 50
BATCH_SIZE = 128  # Tăng từ 64 lên 128 theo Autoformer
LEARNING_RATE = 0.0001

# Biến môi trường MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
MONGO_COLLECTION_PRICE = os.getenv("MONGO_COLLECTION_PRICE")

# Kiểm tra biến môi trường
if not all([MONGO_URI, MONGO_DB, MONGO_COLLECTION, MONGO_COLLECTION_PRICE]):
    missing = [k for k, v in {
        "MONGO_URI": MONGO_URI,
        "MONGO_DB": MONGO_DB,
        "MONGO_COLLECTION": MONGO_COLLECTION,
        "MONGO_COLLECTION_PRICE": MONGO_COLLECTION_PRICE
    }.items() if not v]
    raise ValueError(f"Thiếu các biến môi trường: {', '.join(missing)}")