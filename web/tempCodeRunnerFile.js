const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// Tải biến môi trường từ file .env ở thư mục gốc
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

// Debug: Kiểm tra các biến môi trường
console.log("MONGO_URI:", process.env.MONGO_URI);
console.log("MONGO_DB:", process.env.MONGO_DB);
console.log("MONGO_COLLECTION_PRICE:", process.env.MONGO_COLLECTION_PRICE);
console.log("MONGO_COLLECTION:", process.env.MONGO_COLLECTION);

// Kiểm tra biến môi trường
const requiredEnvVars = ['MONGO_URI', 'MONGO_DB', 'MONGO_COLLECTION_PRICE', 'MONGO_COLLECTION'];
const missingEnvVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingEnvVars.length > 0) {
    console.error(`Thiếu các biến môi trường: ${missingEnvVars.join(', ')}`);
    process.exit(1);
}

// Import routes
const stocksRouter = require('./routes/stocks');
const stockDataRouter = require('./routes/stockData');
const simulateRouter = require('./routes/simulate');

// Import DB utilities
const { connectDB, closeDB } = require('./utils/db');

// Debug: In đường dẫn của thư mục public
const publicPath = path.join(__dirname, 'public');
console.log(`Serving static files from: ${publicPath}`);

// Phục vụ file tĩnh từ thư mục web/public
app.use(express.static(publicPath, {
    setHeaders: (res, path) => {
        console.log(`Serving static file: ${path}`);
        if (path.endsWith('.css')) {
            res.setHeader('Content-Type', 'text/css');
        }
    }
}));

// Route mặc định để phục vụ index.html
app.get('/', (req, res) => {
    console.log("Serving index.html...");
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Định nghĩa routes
app.use('/api/stocks', (req, res, next) => {
    console.log("Request received for /api/stocks");
    next();
}, stocksRouter);

app.use('/api/stock', (req, res, next) => {
    console.log(`Request received for /api/stock/${req.params.symbol}/${req.params.range}`);
    next();
}, stockDataRouter);

app.use('/api/simulate', (req, res, next) => {
    console.log(`Request received for /api/simulate/${req.params.symbol}`);
    next();
}, simulateRouter);

// Xử lý lỗi 404
app.use((req, res) => {
    console.log(`404 Not Found: ${req.url}`);
    res.status(404).send('File not found');
});

// Khởi động server
async function startServer() {
    try {
        const db = await connectDB();
        app.locals.db = db; // Lưu db vào app.locals để các route truy cập

        const server = app.listen(port, () => {
            console.log(`Server chạy tại http://localhost:${port}`);
        });

        // Đóng kết nối MongoDB khi server dừng
        process.on('SIGINT', async () => {
            console.log('Đang đóng server...');
            server.close();
            await closeDB();
            process.exit(0);
        });

        process.on('SIGTERM', async () => {
            console.log('Đang đóng server...');
            server.close();
            await closeDB();
            process.exit(0);
        });
    } catch (err) {
        console.error('Lỗi khởi động server:', err);
        process.exit(1);
    }
}

startServer();