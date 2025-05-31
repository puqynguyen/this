import { updatePriceChart } from './chartConfig.js';

// Cache để lưu trữ dữ liệu
let cache = {};
let maxNewsDate = null; // Lưu ngày cuối cùng có tin tức (maxNewsDate)
let currentStockSymbol = null; // Lưu mã cổ phiếu hiện tại

// Danh sách mã cổ phiếu dự phòng trong trường hợp API thất bại
const fallbackStocks = ['AAPL', 'BID', 'FPT', 'MBB', 'META', 'NVDA', 'VCB'];

// Lấy danh sách cổ phiếu cho dropdown
async function populateDropdown() {
    try {
        console.log("Fetching stock list from /api/stocks...");
        const response = await axios.get('/api/stocks');
        let stocks = response.data;
        console.log(`Stock list received: ${JSON.stringify(stocks)}`);

        // Kiểm tra xem stocks có phải là mảng hợp lệ không
        if (!Array.isArray(stocks) || stocks.length === 0) {
            console.warn("Danh sách mã cổ phiếu không hợp lệ, sử dụng danh sách dự phòng...");
            stocks = fallbackStocks;
        }

        const dropdown = document.getElementById('stockDropdown');
        if (!dropdown) {
            throw new Error("Không tìm thấy phần tử stockDropdown trong DOM");
        }

        dropdown.innerHTML = ''; // Xóa nội dung cũ
        stocks.forEach(stock => {
            const option = document.createElement('option');
            option.value = stock;
            option.textContent = stock;
            dropdown.appendChild(option);
        });

        // Lấy ngày cuối cùng có tin tức (maxNewsDate) cho mã cổ phiếu đầu tiên
        if (stocks.length > 0) {
            currentStockSymbol = stocks[0];
            await populateCurrentDate(stocks[0]);
        }
    } catch (error) {
        console.error('Lỗi khi lấy danh sách mã cổ phiếu:', error);
        // Sử dụng danh sách dự phòng nếu API thất bại
        const dropdown = document.getElementById('stockDropdown');
        if (dropdown) {
            dropdown.innerHTML = ''; // Xóa nội dung cũ
            fallbackStocks.forEach(stock => {
                const option = document.createElement('option');
                option.value = stock;
                option.textContent = stock;
                dropdown.appendChild(option);
            });
        }
        document.getElementById('chartError').textContent = 'Không thể tải danh sách mã cổ phiếu, sử dụng danh sách dự phòng: ' + (error.response?.data?.error || error.message);
    }
}

// Lấy ngày cuối cùng có tin tức (maxNewsDate) và hiển thị
async function populateCurrentDate(stockSymbol) {
    try {
        if (!stockSymbol) {
            throw new Error("stockSymbol không được xác định");
        }
        const response = await axios.get(`/api/stock/${stockSymbol}/news-period`);
        const { minNewsDate, maxNewsDate: maxDate } = response.data;
        console.log(`News period for ${stockSymbol}: from ${minNewsDate} to ${maxDate}`);

        // Lưu maxNewsDate làm mốc để tính ngày dự đoán
        maxNewsDate = new Date(maxDate);
        currentStockSymbol = stockSymbol; // Cập nhật mã cổ phiếu hiện tại

        // Hiển thị ngày cuối cùng có tin tức trên giao diện
        const currentDateElement = document.getElementById('currentDate');
        if (currentDateElement) {
            currentDateElement.textContent = maxNewsDate.toISOString().split('T')[0];
        } else {
            console.error("Không tìm thấy phần tử currentDate trong DOM");
        }

        // Xóa cache khi mã cổ phiếu thay đổi để đảm bảo dữ liệu mới được lấy
        cache = {};
        console.log("Cache cleared due to stock symbol change");
    } catch (error) {
        console.error('Lỗi khi lấy khoảng thời gian có tin tức:', error);
        document.getElementById('chartError').textContent = `Không thể lấy khoảng thời gian có tin tức cho ${stockSymbol}: ` + (error.response?.data?.error || error.message);
        throw error;
    }
}

// Lấy dữ liệu cổ phiếu từ API hoặc cache
async function fetchStockData(stockSymbol, range) {
    if (!stockSymbol || !range) {
        throw new Error('stockSymbol hoặc range không được xác định');
    }
    const cacheKey = `${stockSymbol}-${range}`;
    if (cache[cacheKey]) {
        console.log(`Lấy dữ liệu từ cache cho ${cacheKey}`);
        return cache[cacheKey];
    }

    try {
        console.log(`Fetching data from /api/stock/${stockSymbol}/${range}...`);
        const response = await axios.get(`/api/stock/${stockSymbol}/${range}`);
        const data = response.data;
        console.log(`Data received: ${JSON.stringify(data)}`);

        // Lọc dữ liệu để chỉ hiển thị trong khoảng thời gian có tin tức
        const newsResponse = await axios.get(`/api/stock/${stockSymbol}/news-period`);
        const { minNewsDate, maxNewsDate } = newsResponse.data;
        console.log(`News period for ${stockSymbol}: from ${minNewsDate} to ${maxNewsDate}`);

        const filteredData = {
            dates: [],
            prices: [],
            lastPrice: data.lastPrice,
            change: data.change,
            percentChange: data.percentChange,
            open: data.open,
            high: data.high,
            low: data.low,
            currentDay: new Date(maxNewsDate) // Dùng maxNewsDate để vẽ đường dọc
        };

        for (let i = 0; i < data.dates.length; i++) {
            const date = new Date(data.dates[i]);
            const minDate = new Date(minNewsDate);
            const maxDate = new Date(maxNewsDate);
            if (date >= minDate && date <= maxDate) {
                filteredData.dates.push(data.dates[i]);
                filteredData.prices.push(data.prices[i]);
            }
        }

        console.log(`Filtered data for ${stockSymbol}: ${JSON.stringify(filteredData)}`);
        cache[cacheKey] = filteredData; // Lưu dữ liệu đã lọc vào cache
        document.getElementById('chartError').textContent = '';
        return filteredData;
    } catch (error) {
        console.error('Lỗi khi lấy dữ liệu giá:', error);
        document.getElementById('chartError').textContent = `Không thể tải dữ liệu giá cho ${stockSymbol}: ` + (error.response?.data?.error || error.message);
        throw error;
    }
}

// Dự đoán giá cổ phiếu
async function predictStock(stockSymbol, days) {
    try {
        // Kiểm tra xem stockSymbol có hợp lệ không
        if (!stockSymbol) {
            throw new Error("Mã cổ phiếu không được xác định");
        }

        // Kiểm tra xem days có hợp lệ không
        const daysNum = parseInt(days);
        if (isNaN(daysNum) || daysNum <= 0) {
            throw new Error("Số ngày dự đoán không hợp lệ");
        }

        // Kiểm tra và cập nhật maxNewsDate nếu cần
        if (!maxNewsDate || currentStockSymbol !== stockSymbol) {
            console.log("maxNewsDate chưa được khởi tạo hoặc mã cổ phiếu đã thay đổi, gọi populateCurrentDate...");
            await populateCurrentDate(stockSymbol);
            if (!maxNewsDate) {
                throw new Error("Không thể lấy ngày cuối cùng có tin tức (maxNewsDate)");
            }
        }

        // Tính ngày dự đoán dựa trên maxNewsDate và số ngày người dùng nhập
        const predictDate = new Date(maxNewsDate);
        predictDate.setDate(predictDate.getDate() + daysNum);
        const predictDateStr = predictDate.toISOString().split('T')[0];
        console.log(`Fetching prediction for ${stockSymbol} on ${predictDateStr}...`);
        const response = await axios.get(`/api/stock/${stockSymbol}/predict/${predictDateStr}`);
        const data = response.data;
        console.log(`Prediction received: ${JSON.stringify(data)}`);

        // Cập nhật giao diện
        document.getElementById('predDate').textContent = data.date || 'Không có dữ liệu';
        document.getElementById('predPrice').textContent = data.predictions.price ? data.predictions.price.toFixed(2) : 'Không có dữ liệu';
        document.getElementById('actualPrice').textContent = data.actual.price != null ? data.actual.price.toFixed(2) : 'Không có dữ liệu thực tế';

        document.getElementById('predError').textContent = '';
    } catch (error) {
        console.error('Lỗi khi lấy dữ liệu dự đoán:', error);
        let errorMessage = `Không thể lấy dữ liệu dự đoán cho ${stockSymbol}: ` + (error.response?.data?.error || error.message);
        if (error.response?.data?.error.includes("Ngày")) {
            errorMessage = error.response.data.error;
        } else if (error.response?.data?.error.includes("Không đủ dữ liệu lịch sử")) {
            errorMessage = 'Không đủ dữ liệu lịch sử để dự đoán cho ngày này';
        }
        document.getElementById('predError').textContent = errorMessage;
    }
}

// Cập nhật biểu đồ giá
async function updateChart(stockSymbol, currentTimeRange) {
    console.log(`updateChart called with stockSymbol: ${stockSymbol}, currentTimeRange: ${currentTimeRange}`);
    if (!stockSymbol || !currentTimeRange) {
        console.log("Skipping updateChart: stockSymbol or currentTimeRange is undefined");
        document.getElementById('chartError').textContent = 'Vui lòng chọn mã cổ phiếu và khoảng thời gian';
        return;
    }
    try {
        const data = await fetchStockData(stockSymbol, currentTimeRange);
        console.log(`Updating chart with data: ${JSON.stringify(data)}`);
        updatePriceChart(data.dates, data.prices, data.currentDay);
        document.getElementById('stockPrice').textContent = `${data.lastPrice.toFixed(2)} USD`;
        document.getElementById('stockChange').textContent = `${data.change.toFixed(2)} (${data.percentChange})`;
        document.getElementById('openPrice').textContent = data.open.toFixed(2);
        document.getElementById('highPrice').textContent = data.high.toFixed(2);
        document.getElementById('lowPrice').textContent = data.low.toFixed(2);
        document.getElementById('marketCap').textContent = data.marketCap || '3.09T';
        document.getElementById('peRatio').textContent = data.peRatio || '32.28';
        document.getElementById('divYield').textContent = data.divYield || '0.50%';
        document.getElementById('high52').textContent = data.high52 || '260.09';
        document.getElementById('low52').textContent = data.low52 || '169.21';
    } catch (error) {
        console.error(`Error in updateChart: ${error.message}`);
    }
}

// Export các hàm
export { populateDropdown, fetchStockData, updateChart, predictStock, cache, populateCurrentDate };