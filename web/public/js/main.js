import { populateDropdown, updateChart, predictStock, populateCurrentDate } from './api.js';
import { initializePriceChart } from './chartConfig.js';

// Biến toàn cục
let currentTimeRange = 'MAX';

// Thay đổi khoảng thời gian
function changeTimeRange(range) {
    console.log(`changeTimeRange called with range: ${range}`);
    currentTimeRange = range;
    document.querySelectorAll('.time-buttons button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`#range-${range}`).classList.add('active');
    const stockSymbol = document.getElementById('stockDropdown').value;
    console.log(`Calling updateChart with stockSymbol: ${stockSymbol}, currentTimeRange: ${currentTimeRange}`);
    if (stockSymbol && currentTimeRange) {
        updateChart(stockSymbol, currentTimeRange);
    } else {
        console.log("Skipping updateChart: stockSymbol or currentTimeRange is undefined");
    }
}

// Khởi tạo trang
window.onload = async () => {
    console.log("Window loaded, initializing page...");
    // Khởi tạo biểu đồ
    initializePriceChart();

    // Khởi tạo dropdown và biểu đồ
    try {
        await populateDropdown(); // Đợi populateDropdown hoàn thành để đảm bảo maxNewsDate được khởi tạo
        const stockDropdown = document.getElementById('stockDropdown');
        
        // Đảm bảo dropdown có giá trị mặc định
        if (stockDropdown.options.length > 0 && !stockDropdown.value) {
            stockDropdown.value = stockDropdown.options[0].value; // Chọn giá trị đầu tiên
            console.log(`Default stock selected: ${stockDropdown.value}`);
        }

        // Cập nhật biểu đồ ban đầu
        console.log("Updating chart initially...");
        if (stockDropdown.value) {
            updateChart(stockDropdown.value, currentTimeRange);
        } else {
            console.log("Skipping initial chart update: No stock selected");
        }
    } catch (error) {
        console.error("Error populating dropdown:", error);
    }

    document.getElementById('stockDropdown').addEventListener('change', async () => {
        const stockSymbol = document.getElementById('stockDropdown').value;
        console.log(`Stock dropdown changed to: ${stockSymbol}`);
        if (stockSymbol && currentTimeRange) {
            await populateCurrentDate(stockSymbol); // Cập nhật maxNewsDate khi thay đổi mã cổ phiếu
            updateChart(stockSymbol, currentTimeRange);
        } else {
            console.log("Skipping updateChart: stockSymbol or currentTimeRange is undefined");
        }
    });

    // Thêm sự kiện cho các nút thời gian
    const ranges = ['5D', '1M', '6M', 'YTD', '1Y', '5Y', 'Max'];
    ranges.forEach(range => {
        const button = document.getElementById(`range-${range}`);
        if (button) {
            button.addEventListener('click', () => {
                changeTimeRange(range);
            });
        }
    });

    // Thêm sự kiện cho nút dự đoán
    const predictButton = document.getElementById('predictButton');
    if (predictButton) {
        predictButton.addEventListener('click', async () => {
            const stockSymbol = document.getElementById('stockDropdown').value;
            const days = document.getElementById('daysInput').value;
            console.log(`Predicting for stockSymbol: ${stockSymbol}, days: ${days}`);
            if (stockSymbol && days) {
                await predictStock(stockSymbol, days);
            } else {
                console.log("Skipping prediction: stockSymbol or days is undefined");
                document.getElementById('predError').textContent = 'Vui lòng chọn mã cổ phiếu và số ngày để dự đoán';
            }
        });
    } else {
        console.error("Predict button not found in DOM!");
    }
};