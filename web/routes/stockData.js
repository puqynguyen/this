const express = require('express');
const router = express.Router();
const path = require('path');
const { exec } = require('child_process');
const util = require('util');

// Chuyển exec thành Promise để sử dụng async/await
const execPromise = util.promisify(exec);

// Đường dẫn tuyệt đối đến thư mục data
const DATA_DIR = path.join(__dirname, '..', '..', 'data');

// Cache để lưu trữ dữ liệu
const cache = {};

// API lấy khoảng thời gian có tin tức
router.get('/:symbol/news-period', async (req, res) => {
    const { symbol } = req.params;

    try {
        const db = req.app.locals.db;
        if (!db) {
            throw new Error('MongoDB connection is not established');
        }
        if (!process.env.MONGO_COLLECTION) {
            throw new Error('MONGO_COLLECTION is not defined');
        }
        console.log(`Using collection: ${process.env.MONGO_COLLECTION}`);
        const newsCollection = db.collection(process.env.MONGO_COLLECTION);

        // Kiểm tra mã cổ phiếu có tồn tại không
        const availableStocks = await newsCollection.distinct('stock_symbol');
        console.log(`Available stocks in news collection: ${availableStocks}`);
        if (!availableStocks.includes(symbol)) {
            throw new Error(`Mã cổ phiếu ${symbol} không tồn tại trong dữ liệu tin tức`);
        }

        // Lấy ngày đầu tiên và ngày cuối cùng có tin tức
        const firstNewsRecord = await newsCollection.find({ stock_symbol: symbol })
            .sort({ date: 1 })
            .limit(1)
            .toArray();
        const lastNewsRecord = await newsCollection.find({ stock_symbol: symbol })
            .sort({ date: -1 })
            .limit(1)
            .toArray();

        if (!firstNewsRecord.length || !lastNewsRecord.length) {
            throw new Error(`Không tìm thấy dữ liệu tin tức cho ${symbol}`);
        }

        const minNewsDate = firstNewsRecord[0].date;
        const maxNewsDate = lastNewsRecord[0].date;

        res.json({ minNewsDate, maxNewsDate });
    } catch (error) {
        console.error(`Lỗi khi lấy khoảng thời gian có tin tức: ${error.message}`);
        res.status(404).json({ error: 'Không thể lấy khoảng thời gian có tin tức: ' + error.message });
    }
});

// API dự đoán biến động giá cổ phiếu theo ngày người dùng nhập
router.get('/:symbol/predict/:date', async (req, res) => {
    const { symbol, date } = req.params;

    try {
        // Đường dẫn đến script Python predict.py
        const scriptPath = path.join(__dirname, '..', '..', 'scripts', 'model', 'predict.py');
        console.log(`Calling Python script: ${scriptPath}`);

        // Gọi script Python với tham số stock_symbol và date
        const command = `python "${scriptPath}" --stock ${symbol} --date ${date}`;
        const { stdout, stderr } = await execPromise(command);

        if (stderr) {
            console.error(`Lỗi từ Python script: ${stderr}`);
            throw new Error(stderr);
        }

        // Phân tích kết quả từ Python script
        const result = JSON.parse(stdout);
        if (result.error) {
            throw new Error(result.error);
        }

        res.json(result);
    } catch (error) {
        console.error(`Lỗi khi lấy dữ liệu dự đoán: ${error.message}`);
        res.status(404).json({ error: 'Không thể lấy dữ liệu dự đoán: ' + error.message });
    }
});

// API lấy dữ liệu giá cổ phiếu theo mã và khoảng thời gian
router.get('/:symbol/:range', async (req, res) => {
    const { symbol, range } = req.params;
    const cacheKey = `${symbol}-${range}`;

    // Kiểm tra cache
    if (cache[cacheKey]) {
        console.log(`Lấy dữ liệu từ cache cho ${cacheKey}`);
        return res.json(cache[cacheKey]);
    }

    try {
        const db = req.app.locals.db;
        if (!db) {
            throw new Error('MongoDB connection is not established');
        }
        if (!process.env.MONGO_COLLECTION_PRICE) {
            throw new Error('MONGO_COLLECTION_PRICE is not defined');
        }
        const collection = db.collection(process.env.MONGO_COLLECTION_PRICE);

        // Kiểm tra mã cổ phiếu có tồn tại không
        const availableStocks = await collection.distinct('stock_symbol');
        console.log(`Available stocks: ${availableStocks}`);
        if (!availableStocks.includes(symbol)) {
            throw new Error(`Mã cổ phiếu ${symbol} không tồn tại`);
        }

        // Lấy ngày cuối cùng từ dữ liệu
        const lastRecord = await collection.find({ stock_symbol: symbol })
            .sort({ date: -1 })
            .limit(1)
            .toArray();

        if (lastRecord.length === 0) {
            throw new Error(`Không tìm thấy dữ liệu cho ${symbol}`);
        }
        console.log(`Last record: ${JSON.stringify(lastRecord[0])}`);

        // Chuyển date từ chuỗi thành Date object
        const endDate = new Date(lastRecord[0].date);
        console.log(`End date: ${endDate}`);

        // Xác định khoảng thời gian
        let startDate;
        if (range === '1D') {
            startDate = new Date(endDate);
            startDate.setDate(endDate.getDate() - 1);
        } else if (range === '5D') {
            startDate = new Date(endDate);
            startDate.setDate(endDate.getDate() - 5);
        } else if (range === '1M') {
            startDate = new Date(endDate);
            startDate.setMonth(endDate.getMonth() - 1);
        } else if (range === '6M') {
            startDate = new Date(endDate);
            startDate.setMonth(endDate.getMonth() - 6);
        } else if (range === 'YTD') {
            startDate = new Date(endDate.getFullYear(), 0, 1);
        } else if (range === '1Y') {
            startDate = new Date(endDate);
            startDate.setFullYear(endDate.getFullYear() - 1);
        } else if (range === '5Y') {
            startDate = new Date(endDate);
            startDate.setFullYear(endDate.getFullYear() - 5);
        } else {
            startDate = new Date(0); // Max
        }

        // Chuyển startDate và endDate về định dạng chuỗi YYYY-MM-DD để truy vấn
        const startDateStr = startDate.toISOString().split('T')[0];
        const endDateStr = endDate.toISOString().split('T')[0];
        console.log(`Querying data from ${startDateStr} to ${endDateStr}`);

        // Truy vấn dữ liệu từ MongoDB, chỉ lấy các cột cần thiết
        const data = await collection.find(
            {
                stock_symbol: symbol,
                date: { $gte: startDateStr, $lte: endDateStr }
            },
            {
                projection: {
                    date: 1,
                    last: 1,
                    open: 1,
                    high: 1,
                    low: 1,
                    '%change': 1,
                    _id: 0
                }
            }
        ).sort({ date: 1 }).toArray();

        console.log(`Number of records found: ${data.length}`);
        if (data.length === 0) {
            throw new Error(`Không tìm thấy dữ liệu cho ${symbol} trong khoảng thời gian từ ${startDateStr} đến ${endDateStr}`);
        }

        // Chuẩn bị dữ liệu trả về
        const responseData = {
            dates: data.map(d => d.date),
            prices: data.map(d => parseFloat(d.last)),
            lastPrice: parseFloat(data[data.length - 1]?.last) || 0,
            change: (parseFloat(data[data.length - 1]?.last) - parseFloat(data[0]?.last)) || 0,
            percentChange: typeof data[data.length - 1]?.['%change'] === 'number'
                ? `${data[data.length - 1]['%change'].toFixed(2)}%`
                : data[data.length - 1]?.['%change'] || '0%',
            open: parseFloat(data[data.length - 1]?.open) || 0,
            high: parseFloat(data[data.length - 1]?.high) || 0,
            low: parseFloat(data[data.length - 1]?.low) || 0
        };
        console.log(`Response data: ${JSON.stringify(responseData)}`);

        // Lưu vào cache
        cache[cacheKey] = responseData;
        res.json(responseData);
    } catch (error) {
        console.error(`Lỗi khi lấy dữ liệu giá: ${error.message}`);
        res.status(404).json({ error: 'Không thể lấy dữ liệu giá: ' + error.message });
    }
});

module.exports = router;