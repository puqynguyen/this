let chart;

// Khởi tạo biểu đồ giá
function initializePriceChart() {
    const ctx = document.getElementById('stockChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Giá cổ phiếu',
                data: [],
                borderColor: '#e74c3c',
                fill: false,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: 'Thời gian' },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: { display: true, text: 'Giá (USD)' },
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false
                },
                annotation: {
                    annotations: []
                }
            }
        }
    });
}

// Hàm cập nhật biểu đồ giá với đường dọc màu xanh tại current day
function updatePriceChart(dates, prices, currentDay) {
    chart.data.labels = dates;
    chart.data.datasets[0].data = prices;

    // Tìm vị trí của current day trên trục x
    const currentDayIndex = dates.findIndex(date => date === currentDay.toISOString().split('T')[0]);
    let annotation = {};
    if (currentDayIndex !== -1) {
        annotation = {
            type: 'line',
            scaleID: 'x',
            value: currentDayIndex,
            borderColor: 'green',
            borderWidth: 2,
            label: {
                content: 'Current Day',
                enabled: true,
                position: 'top'
            }
        };
    }

    chart.options.plugins.annotation.annotations = [annotation];
    chart.update();
}

// Export các hàm và biến
export { chart, initializePriceChart, updatePriceChart };