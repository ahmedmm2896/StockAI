import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to fetch stock data from Alpha Vantage API
def fetch_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    # Convert to DataFrame for easy processing
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df = df.sort_index()  # Sort by date
    return df

# Fetch stock data
api_key = "your_api_key_here"
stock_symbol = "AAPL"
df = fetch_stock_data(stock_symbol, api_key)
print(df.head())

# Prepare data for prediction
X = df[['Open', 'High', 'Low', 'Close']].values
y = df['Close'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make a prediction
y_pred = model.predict(X_test)
print(f"Predicted stock price: {y_pred[0]}")

# Plot stock prices
plt.plot(df.index, df['Close'], label='Closing Price')
plt.title(f'Stock Prices of {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
