const express = require('express');
const router = express.Router();

// API lấy danh sách mã cổ phiếu
router.get('/', async (req, res) => {
    try {
        const db = req.app.locals.db;
        const collection = db.collection(process.env.MONGO_COLLECTION_PRICE);
        const stocks = await collection.distinct('stock_symbol');
        res.json(stocks);
    } catch (error) {
        console.error(`Lỗi khi lấy danh sách mã cổ phiếu: ${error.message}`);
        res.status(500).json({ error: 'Không thể lấy danh sách mã cổ phiếu' });
    }
});

module.exports = router;