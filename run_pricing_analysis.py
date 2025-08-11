'''
	This script runs multiple analyses:
	1. Black-Scholes equations & CN PDE estimates for options (pricing_processor.py)
        2. Greeks estimates 
        3. MC simulation of stock price (Geometric Brownian)

        Users can input any features in the BS equation for their own estimation,
	i.e. stock price (S), strike price (K), time to maturity (T), interest rate (r), dividend (q).

	Run general command:
	`python run_pricing_analysis.py`

	Run command with online input (example 'AAPL'):
        `python run_pricing_analysis.py -c AAPL -e 2025-09-05 -K 140 -sg .40 -r 0.01 -y 0.10 -T 0.05 `

'''

import argparse
import yfinance as yf
import datetime
import pandas as pd
from datetime import datetime, timedelta

import pricing_processor
import greeks_processor

def company_option(company, expiry, strike_price):
    stock = yf.Ticker(company)

    stock_history = stock.history(period='1d')
    #stock_history = stock.history(period='7d').dropna(subset=['Close'])
    if stock_history.empty:
        raise ValueError(f"No price data found for {company} with period='1d'.")
    stock_price = stock_history['Close'].iloc[-1]

    # for real option prices
    if company and expiry and strike_price:
        options = stock.option_chain(expiry)
        call = options.calls
        put  = options.puts
        call_price = call[call['strike'] == strike_price] 
        put_price  = put[put['strike'] == strike_price]
    else:
        call_price = None
        put_price  = None
        print("Missing input for company name, expiry or strike price")

    if call_price.empty or put_price.empty:
        if call_price.empty:
            print(f"Available call strikes for {expiry}:")
            print(call['strike'].to_list())
        if put_price.empty:
            print(f"Available put strikes for {expiry}:")
            print(put['strike'].to_list())
        raise ValueError(f"No option found for strike price {strike_price} on {expiry}. Check above list and select available strike using -K option")


    data_1yr = stock.history(period="1y")['Close']
    returns_1yr = data_1yr.pct_change().dropna()
    volatility = returns_1yr.std() * (252**0.5)
    dividend = stock.info.get('dividendYield', 0.0)  

    IRX = yf.Ticker("^IRX")
    IRX_latest = IRX.history(period="1d")['Close'].iloc[-1]
    Interest = IRX_latest / 100 


    return stock_price, call_price['lastPrice'].iloc[0], put_price['lastPrice'].iloc[0], volatility, dividend, interest

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('--stock','-S' , default=100.0, help = 'input stock price')
    parser.add_argument('--time', '-T' , default=1.0, help = 'input time to maturity')
    parser.add_argument('--strike','-K',  type=float, default=100.0, help = 'input strike price')
    parser.add_argument('--interest', '-r' , type=float, default=0.05, help = 'input interest rate')
    parser.add_argument('--volatility', '-sg', type=float, default=0.2, help = 'input volatility')
    parser.add_argument('--dividend', '-y', type=float, default=0.0, help = 'input dividend')
    parser.add_argument('--output','-o'   , default='plots', help = 'Name of the output directory')
    parser.add_argument('--company', '-c', default=None, help = 'input the company name for online info')
    parser.add_argument('--expiry', '-e', default='', help = 'input expiry date')

    args = parser.parse_args()
    stock      = args.stock
    time       = args.time   # in year
    strike     = args.strike
    interest   = args.interest
    volatility = args.volatility
    dividend   = args.dividend
    output     = args.output
    company    = args.company
    expiry     = args.expiry

    #today = pd.Timestamp.now()
    today = datetime.now().date()

    print('\n')
    print(f"Stock Info for {company}")
    if expiry:
       expiry_strp = datetime.strptime(expiry, '%Y-%m-%d').date()

       time = (expiry_strp - today).days / 365
       print("time", time)
    else:
       expiry = today + timedelta(days=365 * time)

    if company:
        stock, call_price, put_price, volatility_est, dividend_company, interest = company_option(company=company, expiry=expiry, strike_price=strike)
        print("Most recent closing stock price:", stock)
        print("call price:", call_price)
        print("put price:", put_price)
        print("volatility:", volatility_est)
        print("interest rate", interest)
        print("dividend yield", dividend_company)

    pricing_processor.run_pricing(S=stock, T=time, K=strike, r=interest, sigma=volatility, y=dividend, output=output)
    greeks_processor.run_Greeks(S=stock, K=strike, T=time, r=interest, sigma=volatility, y=dividend, output=output) 
         
