import pymongo
from config import MONGO_URI, MONGO_DB

# Kết nối MongoDB
def connect_db():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return client, db

# Đóng kết nối MongoDB
def close_db(client):
    client.close()