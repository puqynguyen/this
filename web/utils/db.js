const { MongoClient } = require('mongodb');
const dotenv = require('dotenv');
const path = require('path');

// Tải biến môi trường từ thư mục gốc
dotenv.config({ path: path.join(__dirname, '..', '..', '.env') });

// Kiểm tra MONGO_URI
if (!process.env.MONGO_URI) {
    console.error('Lỗi: MONGO_URI không được định nghĩa trong file .env');
    process.exit(1);
}

const uri = process.env.MONGO_URI;
const client = new MongoClient(uri);
const dbName = process.env.MONGO_DB;

// Kết nối MongoDB
async function connectDB() {
    try {
        await client.connect();
        console.log('Đã kết nối đến MongoDB');
        return client.db(dbName);
    } catch (err) {
        console.error('Lỗi kết nối MongoDB:', err);
        process.exit(1);
    }
}

// Đóng kết nối MongoDB
async function closeDB() {
    await client.close();
    console.log('Đã đóng kết nối MongoDB');
}

module.exports = { connectDB, closeDB, client };