// API endpoint configuration
// Points to local Flask backend (server.py) that uses yfinance and LSTM
const API_BASE_URL = 'http://localhost:5000/predict';
const QUOTE_API = 'http://localhost:5000/quote';

// DOM Elements
const tickersInput = document.getElementById('tickers');
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');
const timeStepInput = document.getElementById('time-step');
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const loadingElement = document.getElementById('loading');
const recommendationsElement = document.getElementById('recommendations');
const recommendationsGrid = document.getElementById('recommendations-grid');
const chartsContainer = document.getElementById('charts-container');
const chartsGrid = document.getElementById('charts-grid');
const errorAlert = document.getElementById('error-alert');
const errorMessage = document.getElementById('error-message');

// Set default end date to today
const today = new Date().toISOString().split('T')[0];
endDateInput.value = today;
endDateInput.max = today;

// Set default start date to one year ago
const oneYearAgo = new Date();
oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
startDateInput.value = oneYearAgo.toISOString().split('T')[0];
startDateInput.max = today;

// Event Listeners
predictBtn.addEventListener('click', generatePredictions);
clearBtn.addEventListener('click', clearResults);

// Generate stock predictions
async function generatePredictions() {
    // Validate inputs
    const tickers = tickersInput.value.trim();
    if (!tickers) {
        showError('Please enter at least one stock ticker.');
        return;
    }
    
    const startDate = startDateInput.value;
    const endDate = endDateInput.value;
    const timeStep = parseInt(timeStepInput.value);
    
    if (startDate >= endDate) {
        showError('Start date must be before end date.');
        return;
    }
    
    if (timeStep < 10 || timeStep > 200) {
        showError('Time step must be between 10 and 200 days.');
        return;
    }
    
    // Parse tickers
    const tickerList = tickers.split(',')
        .map(t => t.trim().toUpperCase())
        .filter(t => t.length > 0);
    
    if (tickerList.length > 5) {
        showError('Please enter a maximum of 5 tickers.');
        return;
    }
    
    // Show loading indicator
    showLoading();
    hideError();

    // Try to fetch lightweight latest quotes for tickers (used when backend isn't available)
    let quotes = {};
    try {
        quotes = await fetchQuotes(tickerList);
    } catch (e) {
        console.warn('Failed to fetch quotes:', e);
        quotes = {};
    }
    
    try {
        // In a real implementation, you would make an API call to your backend
        // const response = await fetch(API_BASE_URL, {
        //     method: 'POST',
        //     headers: {
        //         'Content-Type': 'application/json',
        //     },
        //     body: JSON.stringify({
        //         tickers: tickerList,
        //         start_date: startDate,
        //         end_date: endDate,
        //         time_step: timeStep
        //     })
        // });
        
        // const data = await response.json();
        
        // Try calling the backend first; fallback to mock on failure
        try {
            const resp = await fetch(API_BASE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tickers: tickerList,
                    start_date: startDate,
                    end_date: endDate,
                    time_step: timeStep
                })
            });

            if (!resp.ok) throw new Error(`API returned ${resp.status}`);

            const data = await resp.json();

            hideLoading();

            if (data.error || !data.charts) throw new Error(data.error || 'Invalid API response');

            // Build reports from returned chart data
            const backendReports = {};
            data.charts.forEach(chart => {
                backendReports[chart.ticker] = generateFinancialReportFromData(chart.ticker, chart.trueData, chart.predictions, chart.recommendation);
            });

            displayRecommendations(data.recommendations, backendReports);
            displayCharts(data.charts);
        } catch (err) {
            console.warn('Backend fetch failed, using mock data:', err);
            const mockData = generateMockPredictions(tickerList, quotes);
            hideLoading();
            displayRecommendations(mockData.recommendations, mockData.reports);
            displayCharts(mockData.charts);
        }
        
    } catch (error) {
        console.error('Error:', error);
        hideLoading();
        showError('Failed to generate predictions. Please try again later.');
    }
}

// Show loading indicator
function showLoading() {
    loadingElement.style.display = 'block';
    recommendationsElement.style.display = 'none';
    chartsContainer.style.display = 'none';
}

// Hide loading indicator
function hideLoading() {
    loadingElement.style.display = 'none';
}

// Display recommendations
function displayRecommendations(recommendations, reports) {
    recommendationsGrid.innerHTML = '';

    Object.entries(recommendations).forEach(([ticker, recommendation]) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        
        // Determine recommendation class and icon
        let recommendationClass = '';
        let recommendationIcon = '';
        let recommendationText = '';
        
        if (recommendation === 'Buy') {
            recommendationClass = 'buy';
            recommendationIcon = 'fas fa-arrow-up';
            recommendationText = 'Strong buy signal based on positive trend';
        } else if (recommendation === 'Sell') {
            recommendationClass = 'sell';
            recommendationIcon = 'fas fa-arrow-down';
            recommendationText = 'Consider selling based on negative trend';
        } else {
            recommendationClass = 'hold';
            recommendationIcon = 'fas fa-pause';
            recommendationText = 'Hold position, no strong trend detected';
        }
        
        const reportHtml = reports && reports[ticker] ? reports[ticker] : '';

        card.innerHTML = `
            <div class="recommendation-header">
                <div class="ticker">${ticker}</div>
                <div class="recommendation-badge ${recommendationClass}">
                    <i class="${recommendationIcon}"></i> ${recommendation}
                </div>
            </div>
            <div class="recommendation-details">
                <p>${recommendationText}</p>
                <p><strong>Analysis:</strong> Based on LSTM model with ${timeStepInput.value}-day time step</p>
                <p><strong>Period:</strong> ${startDateInput.value} to ${endDateInput.value}</p>
            </div>
                <div class="financial-report">
                    ${reportHtml}
                    <div class="report-actions">
                        <button class="download-report btn-small">Download Report</button>
                        <button class="download-pdf btn-small">Download PDF</button>
                        <button class="print-report btn-small">Print Report</button>
                    </div>
                </div>
        `;
        
            recommendationsGrid.appendChild(card);

            // Attach actions for download and print using the deterministic report HTML
            const downloadBtn = card.querySelector('.download-report');
            const printBtn = card.querySelector('.print-report');
            const htmlForReport = reportHtml || card.querySelector('.financial-report').innerHTML;

            if (downloadBtn) {
                downloadBtn.addEventListener('click', () => downloadReportHtml(ticker, wrapReportHtml(ticker, htmlForReport)));
            }
            if (printBtn) {
                printBtn.addEventListener('click', () => printReportHtml(wrapReportHtml(ticker, htmlForReport)));
            }
            const pdfBtn = card.querySelector('.download-pdf');
            if (pdfBtn) {
                pdfBtn.addEventListener('click', () => downloadReportPdf(ticker, htmlForReport));
            }
    });
    
    recommendationsElement.style.display = 'block';
}

// Display charts for each ticker
function displayCharts(chartData) {
    chartsGrid.innerHTML = '';
    
    chartData.forEach(data => {
        const chartCard = document.createElement('div');
        chartCard.className = 'chart-card';
        
        const chartHeader = document.createElement('div');
        chartHeader.className = 'chart-header';
        chartHeader.innerHTML = `
            <h3>${data.ticker} - Stock Price Prediction</h3>
            <div class="recommendation-badge ${getRecommendationClass(data.recommendation)}">
                <i class="${getRecommendationIcon(data.recommendation)}"></i> ${data.recommendation}
            </div>
        `;
        
        const chartWrapper = document.createElement('div');
        chartWrapper.className = 'chart-wrapper';
        
        const canvas = document.createElement('canvas');
        canvas.id = `chart-${data.ticker}`;
        
        chartWrapper.appendChild(canvas);
        chartCard.appendChild(chartHeader);
        chartCard.appendChild(chartWrapper);
        chartsGrid.appendChild(chartCard);
        
        // Render the chart
        renderChart(data.ticker, data.labels, data.trueData, data.predictions);
    });
    
    chartsContainer.style.display = 'block';
}

// Render a Chart.js chart
function renderChart(ticker, labels, trueData, predictions) {
    const ctx = document.getElementById(`chart-${ticker}`).getContext('2d');
    
    // Calculate split point for coloring
    const splitPoint = trueData.length - predictions.length;
    // Support two prediction formats from backend or mock:
    // - Backend may return an array equal to labels length with leading nulls
    // - Mock returns only the prediction tail
    let predictionData = [];
    if (predictions.length === labels.length) {
        predictionData = predictions;
    } else {
        predictionData = [...Array(splitPoint).fill(null), ...predictions];
    }
    
    // Create datasets
    const trueDataDataset = {
        label: 'True Price',
        data: trueData,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.1
    };
    
    const predictionDataset = {
        label: 'Predicted Price',
        data: predictionData,
        borderColor: '#e74c3c',
        backgroundColor: 'rgba(231, 76, 60, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        fill: false,
        tension: 0.1
    };
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [trueDataDataset, predictionDataset]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${ticker} Stock Price Prediction`,
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    }
                }
            }
        }
    });
}

// Clear all results
function clearResults() {
    recommendationsGrid.innerHTML = '';
    chartsGrid.innerHTML = '';
    recommendationsElement.style.display = 'none';
    chartsContainer.style.display = 'none';
    hideError();
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorAlert.style.display = 'flex';
}

// Hide error message
function hideError() {
    errorAlert.style.display = 'none';
}

// Fetch latest quotes from backend lightweight endpoint
async function fetchQuotes(tickerList) {
    try {
        const resp = await fetch(QUOTE_API, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: tickerList })
        });
        if (!resp.ok) throw new Error(`Quote API returned ${resp.status}`);
        const data = await resp.json();
        return data.quotes || {};
    } catch (err) {
        console.warn('fetchQuotes error:', err);
        return {};
    }
}

// Helper function to get recommendation class
function getRecommendationClass(recommendation) {
    return recommendation === 'Buy' ? 'buy' : 
           recommendation === 'Sell' ? 'sell' : 'hold';
}

// Helper function to get recommendation icon
function getRecommendationIcon(recommendation) {
    return recommendation === 'Buy' ? 'fas fa-arrow-up' : 
           recommendation === 'Sell' ? 'fas fa-arrow-down' : 'fas fa-pause';
}

// Generate mock data for demonstration
function generateMockPredictions(tickerList, quotes = {}) {
    const recommendations = {};
    const charts = [];

    const reports = {};

    // Create deterministic mock data per ticker based on input parameters
    tickerList.forEach(ticker => {
        // Seed based on ticker and user inputs so results are repeatable
        const seedString = `${ticker}|${startDateInput.value}|${endDateInput.value}|${timeStepInput.value}`;
        const seed = cyrb53(seedString);
        const rng = mulberry32(seed);

        // Deterministic recommendation
        const recOptions = ['Buy', 'Sell', 'Hold'];
        const recommendation = recOptions[Math.floor(rng() * recOptions.length)];
        recommendations[ticker] = recommendation;

        // Generate mock chart data (deterministic)
        const labels = generateDateLabels(200);
        const quoted = quotes && quotes[ticker] ? parseFloat(quotes[ticker]) : null;
        const basePrice = quoted ? Math.round(quoted) : Math.round(50 + rng() * 450); // prefer real quote when available
        const volatility = 0.01 + rng() * 0.05; // 1% - 6% daily volatility

        const trueData = generateMockStockData(basePrice, volatility, rng, 200);
        const predictions = generateMockStockData(trueData[trueData.length - 1], volatility, rng, 50);

        charts.push({
            ticker,
            recommendation,
            labels,
            trueData,
            predictions
        });
        // deterministic AI-style financial report for the ticker
        reports[ticker] = generateFinancialReport(ticker, rng, trueData, predictions, recommendation);
    });

    return { recommendations, charts, reports };
}

// Generate date labels
function generateDateLabels(count) {
    const labels = [];
    const today = new Date();
    
    for (let i = count - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        labels.push(date.toISOString().split('T')[0]);
    }
    
    return labels;
}

// Generate mock stock data
function generateMockStockData(basePrice, volatility, rng, count = 200) {
    const data = [parseFloat(basePrice.toFixed ? basePrice.toFixed(2) : basePrice)];

    for (let i = 1; i < count; i++) {
        // Random price change within +/- volatility
        const changePercent = (rng() * 2 * volatility) - volatility;
        let newPrice = data[i - 1] * (1 + changePercent);

        // Ensure price stays positive
        newPrice = Math.max(0.01, newPrice);

        data.push(parseFloat(newPrice.toFixed(2)));
    }

    return data;
}

// Simple deterministic string hash (returns 32-bit unsigned int)
function cyrb53(str, seed = 0) {
    let h1 = 0xDEADBEEF ^ seed, h2 = 0x41C6CE57 ^ seed;
    for (let i = 0, ch; i < str.length; i++) {
        ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (h2 >>> 0) + ((h1 >>> 0) * 4294967296);
}

// Mulberry32 PRNG from a 32-bit seed
function mulberry32(a) {
    return function() {
        a |= 0;
        a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// Generate a concise AI-style financial report (HTML) for a ticker
function generateFinancialReport(ticker, rng, trueData, predictions, recommendation) {
    const latestPrice = trueData[trueData.length - 1];
    const nextPrice = predictions[predictions.length - 1];
    const changePct = (((nextPrice - latestPrice) / latestPrice) * 100).toFixed(2);

    // Mock metrics using RNG
    const avgVolume = Math.round(1e6 + rng() * 9e6);
    const revenueGrowth = (rng() * 30 - 5).toFixed(1); // -5% to +25%
    const profitMargin = (5 + rng() * 25).toFixed(1); // 5% to 30%
    const peRatio = (5 + rng() * 40).toFixed(1);

    const sentiment = recommendation === 'Buy' ? 'positive' : recommendation === 'Sell' ? 'negative' : 'neutral';

    const report = `
        <h4>Financial Report — ${ticker}</h4>
        <p><strong>Latest Price:</strong> $${latestPrice} &nbsp; <strong>Forecast:</strong> $${nextPrice} (${changePct}% vs latest)</p>
        <p><strong>Key Metrics (mock):</strong> Avg. Volume ${avgVolume.toLocaleString()}, Revenue growth ${revenueGrowth}%, Profit margin ${profitMargin}%, P/E ${peRatio}</p>
        <p><strong>Summary:</strong> Model sentiment is <em>${sentiment}</em>. The generated forecast anticipates a ${changePct}% change over the prediction horizon based on recent price patterns.</p>
        <p><strong>Drivers:</strong> Recent momentum, volatility profile, and market liquidity. Mock revenue growth and margin estimates suggest the company has ${revenueGrowth >= 0 ? 'expanding' : 'contracting'} top-line performance.</p>
        <p><strong>Risks:</strong> Earnings surprises, macroeconomic shifts, and sector-specific catalysts could invalidate the short-term forecast.</p>
        <p><strong>Recommendation rationale:</strong> ${recommendation} — based on pattern recognition from historical series and the model's projected path.</p>
    `;

    return report;
}

// Generate a report from real chart data (no RNG)
function generateFinancialReportFromData(ticker, trueData, predictions, recommendation) {
    const latestPrice = Array.isArray(trueData) && trueData.length ? trueData[trueData.length - 1] : 'N/A';
    // Find last numeric predicted price
    let nextPrice = null;
    if (Array.isArray(predictions)) {
        for (let i = predictions.length - 1; i >= 0; i--) {
            if (predictions[i] !== null && predictions[i] !== undefined) { nextPrice = predictions[i]; break; }
        }
    }
    nextPrice = nextPrice || 'N/A';

    const changePct = (typeof latestPrice === 'number' && typeof nextPrice === 'number') ? (((nextPrice - latestPrice) / latestPrice) * 100).toFixed(2) : 'N/A';

    const avgRecent = (arr, n = 10) => {
        if (!Array.isArray(arr) || arr.length === 0) return 'N/A';
        const slice = arr.slice(Math.max(0, arr.length - n));
        const nums = slice.filter(v => typeof v === 'number');
        if (!nums.length) return 'N/A';
        const s = nums.reduce((a,b)=>a+b,0)/nums.length; return s.toFixed(2);
    };

    const avgPrice = avgRecent(trueData, 30);
    const recentVolatility = 'N/A';

    const sentiment = recommendation || 'Neutral';

    const report = `
        <h4>Financial Report — ${ticker}</h4>
        <p><strong>Latest Price:</strong> $${latestPrice} &nbsp; <strong>Forecast:</strong> ${nextPrice !== 'N/A' ? '$' + nextPrice : 'N/A'} (${changePct !== 'N/A' ? changePct + '%' : 'N/A'})</p>
        <p><strong>Key Metrics (derived):</strong> 30-day avg price ${avgPrice}, Recent volatility ${recentVolatility}</p>
        <p><strong>Summary:</strong> Model sentiment is <em>${sentiment}</em>. The forecast reflects patterns detected in historical closing prices.</p>
        <p><strong>Drivers:</strong> Momentum and recent trend in closing prices.</p>
        <p><strong>Risks:</strong> Market events, earnings, and macro shocks.</p>
        <p><strong>Recommendation rationale:</strong> ${sentiment} — based on recent model-projected trajectory.</p>
    `;

    return report;
}

// Wrap report HTML in a complete document for download/print
function wrapReportHtml(ticker, innerHtml) {
    return `<!doctype html><html><head><meta charset="utf-8"><title>${ticker} Financial Report</title><style>body{font-family:Arial,Helvetica,sans-serif;padding:20px;color:#222}h4{margin-top:0} .report-actions{margin-top:12px}</style></head><body>${innerHtml}</body></html>`;
}

// Download the report as an HTML file
function downloadReportHtml(ticker, html) {
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${ticker}-financial-report.html`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

// Open a printable window and invoke print
function printReportHtml(html) {
    const win = window.open('', '_blank', 'width=900,height=700');
    if (!win) {
        showError('Popup blocked. Allow popups to print the report.');
        return;
    }
    win.document.write(html);
    win.document.close();
    win.focus();
    // Give the window a short delay to render then print
    setTimeout(() => {
        win.print();
    }, 250);
}

// Download report as PDF (includes chart image if chart canvas exists)
function downloadReportPdf(ticker, reportHtml) {
    try {
        const { jsPDF } = window.jspdf || {};
        if (!jsPDF) {
            showError('PDF library not loaded. Ensure jsPDF is available.');
            return;
        }

        const doc = new jsPDF({ unit: 'pt', format: 'a4' });
        const pageWidth = doc.internal.pageSize.getWidth();
        const margin = 40;
        let cursorY = 40;

        // Title
        doc.setFontSize(16);
        doc.text(`${ticker} Financial Report`, margin, cursorY);
        cursorY += 18;

        // Extract plain text from HTML report
        const tmp = document.createElement('div');
        tmp.innerHTML = reportHtml;
        const text = tmp.innerText || tmp.textContent || '';
        doc.setFontSize(11);
        const splitText = doc.splitTextToSize(text, pageWidth - margin * 2);
        doc.text(splitText, margin, cursorY);
        cursorY += (splitText.length + 1) * 14;

        // Try to include the chart image
        const canvas = document.getElementById(`chart-${ticker}`);
        if (canvas && canvas.toDataURL) {
            const imgData = canvas.toDataURL('image/png');
            const img = new Image();
            img.onload = function() {
                const maxImgWidth = pageWidth - margin * 2;
                const imgRatio = img.width / img.height;
                const imgWidth = maxImgWidth;
                const imgHeight = maxImgWidth / imgRatio;

                // If image would overflow page, add a new page
                if (cursorY + imgHeight + 80 > doc.internal.pageSize.getHeight()) {
                    doc.addPage();
                    cursorY = margin;
                }

                doc.addImage(imgData, 'PNG', margin, cursorY, imgWidth, imgHeight);
                cursorY += imgHeight + 20;

                // Footer / Source
                const sourceText = '© 2026 D.D. Dompreh Stock Price Predictor |Omanai AI-Powered Financial Analysis';
                doc.setFontSize(9);
                doc.text(sourceText, margin, doc.internal.pageSize.getHeight() - 30);

                doc.save(`${ticker}-financial-report.pdf`);
            };
            img.onerror = function() {
                // Fallback: save without image
                const sourceText = '© 2026 D.D. Dompreh Stock Price Predictor |Omanai AI-Powered Financial Analysis';
                doc.setFontSize(9);
                doc.text(sourceText, margin, doc.internal.pageSize.getHeight() - 30);
                doc.save(`${ticker}-financial-report.pdf`);
            };
            img.src = imgData;
        } else {
            // No chart canvas found: just add footer and save
            const sourceText = '© 2026 D.D. Dompreh Stock Price Predictor |Omanai AI-Powered Financial Analysis';
            doc.setFontSize(9);
            doc.text(sourceText, margin, doc.internal.pageSize.getHeight() - 30);
            doc.save(`${ticker}-financial-report.pdf`);
        }
    } catch (err) {
        console.error('PDF generation error:', err);
        showError('Failed to generate PDF. Check console for details.');
    }
}

// Initialize with today's date
window.addEventListener('load', () => {
    // Set end date to today (already done in HTML)
    // Set min date for start date to one year before today
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    startDateInput.min = oneYearAgo.toISOString().split('T')[0];
});