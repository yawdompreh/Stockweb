"""
This is a simplified backend server that would run your prediction code.
In production, you would need to set up proper authentication, error handling, and scaling.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple, Optional
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def get_ticker_info(ticker: str) -> Dict:
    """Get comprehensive ticker information from yfinance"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Get current price with priority
        current_price = None
        price_fields = ['regularMarketPrice', 'currentPrice', 'ask', 'bid', 'previousClose']
        
        for field in price_fields:
            if field in info and info[field] is not None:
                current_price = float(info[field])
                break
        
        # Fallback to history
        if current_price is None:
            hist = t.history(period='1d')
            if not hist.empty and 'Close' in hist.columns:
                current_price = float(hist['Close'].iloc[-1])
        
        # Get financial metrics
        financials = {
            'marketCap': info.get('marketCap'),
            'volume': info.get('volume'),
            'averageVolume': info.get('averageVolume'),
            'avgVolume10days': info.get('averageVolume10days'),
            'peRatio': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'profitMargins': info.get('profitMargins'),
            'operatingMargins': info.get('operatingMargins'),
            'revenueGrowth': info.get('revenueGrowth'),
            'earningsGrowth': info.get('earningsGrowth'),
            'dividendYield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            'fiftyDayAverage': info.get('fiftyDayAverage'),
            'twoHundredDayAverage': info.get('twoHundredDayAverage')
        }
        
        # Clean None values
        financials = {k: v for k, v in financials.items() if v is not None}
        
        return {
            'current_price': current_price,
            'currency': info.get('currency', 'USD'),
            'name': info.get('shortName', ticker),
            'symbol': info.get('symbol', ticker),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'financials': financials,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error getting ticker info for {ticker}: {str(e)}")
        return None

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch stock data with validation"""
    try:
        # Add one day to end date to ensure we get data up to the end date
        end_date = datetime.strptime(end, '%Y-%m-%d')
        end_date_plus_one = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start, end=end_date_plus_one, progress=False)
        
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
            
        # Use Adjusted Close if available, otherwise use Close
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        else:
            price_data = data['Close']
            
        return price_data.dropna()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        raise

def create_dataset(dataset: np.ndarray, time_step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training"""
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_model(time_step: int) -> Sequential:
    """Build LSTM model architecture"""
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_analysis_report(ticker: str, 
                            ticker_info: Dict, 
                            recommendation: str, 
                            latest_price: float, 
                            forecast_price: float,
                            trend_pct: float,
                            data_points: int,
                            start_date: str,
                            end_date: str,
                            time_step: int) -> Dict:
    """Generate comprehensive analysis report with actual financial data"""
    
    # Calculate forecast metrics
    price_change = forecast_price - latest_price
    price_change_pct = trend_pct
    
    # Determine confidence level based on trend strength
    if abs(trend_pct) > 3:
        confidence = "High"
    elif abs(trend_pct) > 1.5:
        confidence = "Moderate"
    else:
        confidence = "Low"
    
    # Get financial metrics from ticker info
    financials = ticker_info.get('financials', {})
    
    # Generate risk assessment
    risks = []
    if financials.get('beta', 1.0) > 1.5:
        risks.append("High volatility (Beta > 1.5)")
    if financials.get('peRatio', 0) > 30:
        risks.append("High valuation (P/E > 30)")
    if financials.get('profitMargins', 0) < 0.1:
        risks.append("Low profit margins")
    if not risks:
        risks.append("Moderate market risk")
    
    # Generate opportunities
    opportunities = []
    if financials.get('revenueGrowth', 0) > 0.15:
        opportunities.append(f"Strong revenue growth ({financials.get('revenueGrowth', 0)*100:.1f}%)")
    if financials.get('profitMargins', 0) > 0.2:
        opportunities.append(f"High profit margins ({financials.get('profitMargins', 0)*100:.1f}%)")
    if latest_price < financials.get('fiftyDayAverage', latest_price * 1.1):
        opportunities.append(f"Trading below 50-day average (${financials.get('fiftyDayAverage', 0):.2f})")
    if not opportunities:
        opportunities.append("Market average performance")
    
    # Generate recommendation rationale
    rationale = ""
    if recommendation in ["Strong Buy", "Buy"]:
        rationale = f"Positive momentum with {price_change_pct:.2f}% projected gain. "
        if opportunities:
            rationale += f"Key opportunities: {', '.join(opportunities[:2])}. "
    elif recommendation in ["Strong Sell", "Sell"]:
        rationale = f"Negative momentum with {abs(price_change_pct):.2f}% projected decline. "
        if risks:
            rationale += f"Key risks: {', '.join(risks[:2])}. "
    else:
        rationale = f"Neutral outlook with minimal projected change ({price_change_pct:.2f}%). "
    
    rationale += f"Based on LSTM analysis of {data_points} data points with {time_step}-day lookback period."
    
    # Format large numbers
    def format_number(num):
        if num >= 1e9:
            return f"${num/1e9:.1f}B"
        elif num >= 1e6:
            return f"${num/1e6:.1f}M"
        elif num >= 1e3:
            return f"${num/1e3:.1f}K"
        else:
            return f"${num:.0f}"
    
    # Build the report
    report = {
        'ticker': ticker,
        'company_name': ticker_info.get('name', ticker),
        'sector': ticker_info.get('sector', 'N/A'),
        'industry': ticker_info.get('industry', 'N/A'),
        'recommendation': recommendation,
        'confidence': confidence,
        'analysis_period': f"{start_date} to {end_date}",
        'model_info': f"LSTM model with {time_step}-day time step, trained on {data_points} data points",
        
        'price_analysis': {
            'latest_price': f"${latest_price:.2f}",
            'forecast_price': f"${forecast_price:.2f}",
            'price_change': f"${price_change:.2f}",
            'price_change_pct': f"{price_change_pct:.2f}%",
            'price_vs_52w_high': f"{(latest_price / financials.get('fiftyTwoWeekHigh', latest_price) - 1) * 100:.1f}%" if financials.get('fiftyTwoWeekHigh') else "N/A",
            'price_vs_50d_avg': f"{(latest_price / financials.get('fiftyDayAverage', latest_price) - 1) * 100:.1f}%" if financials.get('fiftyDayAverage') else "N/A"
        },
        
        'financial_metrics': {
            'market_cap': format_number(financials.get('marketCap', 0)),
            'avg_volume': format_number(financials.get('averageVolume', financials.get('avgVolume10days', 0))),
            'pe_ratio': f"{financials.get('peRatio', 0):.1f}",
            'forward_pe': f"{financials.get('forwardPE', 0):.1f}" if financials.get('forwardPE') else "N/A",
            'profit_margin': f"{financials.get('profitMargins', 0)*100:.1f}%" if financials.get('profitMargins') else "N/A",
            'revenue_growth': f"{financials.get('revenueGrowth', 0)*100:.1f}%" if financials.get('revenueGrowth') else "N/A",
            'beta': f"{financials.get('beta', 1.0):.2f}",
            'dividend_yield': f"{financials.get('dividendYield', 0)*100:.2f}%" if financials.get('dividendYield') else "0%"
        },
        
        'technical_metrics': {
            '52_week_high': f"${financials.get('fiftyTwoWeekHigh', 0):.2f}" if financials.get('fiftyTwoWeekHigh') else "N/A",
            '52_week_low': f"${financials.get('fiftyTwoWeekLow', 0):.2f}" if financials.get('fiftyTwoWeekLow') else "N/A",
            '50_day_avg': f"${financials.get('fiftyDayAverage', 0):.2f}" if financials.get('fiftyDayAverage') else "N/A",
            '200_day_avg': f"${financials.get('twoHundredDayAverage', 0):.2f}" if financials.get('twoHundredDayAverage') else "N/A"
        },
        
        'risk_assessment': risks,
        'opportunities': opportunities,
        'summary': rationale,
        
        'model_notes': [
            "Forecast based on historical price patterns using LSTM neural network",
            "Short-term projection (1-5 days ahead)",
            "Does not account for breaking news or macroeconomic events",
            "Past performance does not guarantee future results"
        ],
        
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        time_step = data.get('time_step', 100)
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Limit to 3 tickers for performance
        tickers = tickers[:3]
        
        results = {
            'recommendations': {},
            'charts': [],
            'analysis_reports': {},
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'time_step': time_step,
                'model': 'LSTM Neural Network',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        for ticker in tickers:
            try:
                print(f"Processing {ticker}...")
                
                # Get ticker information with financial metrics
                ticker_info = get_ticker_info(ticker)
                if not ticker_info:
                    results['recommendations'][ticker] = 'Unable to fetch ticker information'
                    continue
                
                latest_price = ticker_info['current_price']
                
                # Fetch historical data
                stock_data = fetch_stock_data(ticker, start_date, end_date)
                
                if len(stock_data) < time_step + 50:
                    results['recommendations'][ticker] = 'Insufficient Data'
                    continue
                
                # Scale data
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(np.array(stock_data).reshape(-1, 1))
                
                # Split data (80% train, 20% test)
                training_size = int(len(data_scaled) * 0.8)
                train_data = data_scaled[:training_size]
                test_data = data_scaled[training_size:]
                
                # Create datasets
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)
                
                if len(X_test) == 0:
                    results['recommendations'][ticker] = 'Insufficient Test Data'
                    continue
                
                # Reshape for LSTM
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                # Build and train model
                model = build_model(time_step)
                model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0, validation_split=0.1)
                
                # Make predictions
                test_predict = model.predict(X_test, verbose=0)
                test_predict = scaler.inverse_transform(test_predict)
                
                # Generate future predictions (next 5 days)
                last_sequence = data_scaled[-time_step:].reshape(1, time_step, 1)
                future_predictions = []
                
                for _ in range(5):  # Predict next 5 days
                    next_pred = model.predict(last_sequence, verbose=0)
                    future_predictions.append(float(scaler.inverse_transform(next_pred)[0, 0]))
                    
                    # Update sequence for next prediction
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = next_pred[0, 0]
                
                # Calculate trend based on recent performance and future predictions
                recent_avg = np.mean(test_predict[-10:]) if len(test_predict) >= 10 else test_predict[-1]
                forecast_price = future_predictions[0]  # Next day prediction
                
                trend = forecast_price - latest_price
                trend_pct = (trend / latest_price) * 100
                
                # Determine recommendation
                if trend_pct > 3:
                    recommendation = 'Strong Buy'
                elif trend_pct > 1:
                    recommendation = 'Buy'
                elif trend_pct < -3:
                    recommendation = 'Strong Sell'
                elif trend_pct < -1:
                    recommendation = 'Sell'
                else:
                    recommendation = 'Hold'
                
                # Generate comprehensive analysis report
                analysis_report = generate_analysis_report(
                    ticker=ticker,
                    ticker_info=ticker_info,
                    recommendation=recommendation,
                    latest_price=latest_price,
                    forecast_price=forecast_price,
                    trend_pct=trend_pct,
                    data_points=len(stock_data),
                    start_date=start_date,
                    end_date=end_date,
                    time_step=time_step
                )
                
                results['analysis_reports'][ticker] = analysis_report
                results['recommendations'][ticker] = recommendation
                
                # Prepare chart data
                true_prices = scaler.inverse_transform(data_scaled).flatten().tolist()
                
                # Historical predictions (aligned with test data)
                historical_preds = [None] * (time_step + 1 + training_size)
                historical_preds.extend(test_predict.flatten().tolist())
                
                # Future predictions
                future_dates = []
                last_date = stock_data.index[-1]
                for i in range(1, 6):
                    future_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
                
                dates = stock_data.index.strftime('%Y-%m-%d').tolist() + future_dates
                all_true_prices = true_prices + [None] * 5
                all_predictions = historical_preds[:len(true_prices)] + future_predictions
                
                # Ensure equal lengths
                min_len = min(len(dates), len(all_true_prices), len(all_predictions))
                dates = dates[:min_len]
                all_true_prices = all_true_prices[:min_len]
                all_predictions = all_predictions[:min_len]
                
                results['charts'].append({
                    'ticker': ticker,
                    'recommendation': recommendation,
                    'labels': dates,
                    'trueData': all_true_prices,
                    'predictions': all_predictions,
                    'latest_price': latest_price,
                    'forecast_price': forecast_price,
                    'future_predictions': future_predictions,
                    'future_dates': future_dates
                })
                
                print(f"Completed {ticker}: {recommendation} ({trend_pct:.2f}%)")
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                traceback.print_exc()
                results['recommendations'][ticker] = f'Error: {str(e)}'
        
        return jsonify(results)
    
    except Exception as e:
        print(f"General error in predict: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/quote', methods=['POST'])
def quote():
    try:
        data = request.get_json() or {}
        tickers = data.get('tickers') or []

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        quotes = {}
        for ticker in tickers:
            try:
                info = get_ticker_info(ticker)
                if info:
                    quotes[ticker] = info
                else:
                    quotes[ticker] = {'error': 'Unable to fetch quote'}
            except Exception as e:
                print(f"Error fetching quote for {ticker}: {str(e)}")
                quotes[ticker] = {'error': str(e)}

        return jsonify({'quotes': quotes})
    except Exception as e:
        print(f"General error in quote: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-quote', methods=['POST'])
def batch_quote():
    """Batch quote endpoint for multiple tickers with efficient processing"""
    try:
        data = request.get_json() or {}
        tickers = data.get('tickers') or []

        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Limit to 20 tickers
        tickers = tickers[:20]
        
        # Use yfinance's batch download for efficiency
        try:
            # Download current day data
            data = yf.download(' '.join(tickers), period='1d', group_by='ticker', progress=False)
            
            quotes = {}
            for ticker in tickers:
                try:
                    if not data.empty and ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker]
                        if not ticker_data.empty:
                            last_row = ticker_data.iloc[-1]
                            quotes[ticker] = {
                                'price': float(last_row['Close']),
                                'open': float(last_row['Open']),
                                'high': float(last_row['High']),
                                'low': float(last_row['Low']),
                                'volume': int(last_row['Volume']) if not pd.isna(last_row['Volume']) else 0,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'batch_download'
                            }
                    else:
                        # Fallback to individual
                        t = yf.Ticker(ticker)
                        info = t.info
                        quotes[ticker] = {
                            'price': info.get('regularMarketPrice', info.get('currentPrice', info.get('previousClose'))),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'individual_fallback'
                        }
                except Exception as e:
                    quotes[ticker] = {'error': str(e), 'timestamp': datetime.now().isoformat()}
            
            return jsonify({'quotes': quotes})
            
        except Exception as e:
            return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.2.0',
        'endpoints': ['/predict', '/quote', '/batch-quote', '/health']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')