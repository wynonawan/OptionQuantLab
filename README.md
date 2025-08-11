# OptionQuantLab
This quant lab repo is built for exploring and visualizing various financial theories, as well as observing how they are applied to real market parameters.
It implements multiple calculations for options, including Black-Scholes equations, PDE and Greeks.
BS equations here are typically used for European options with non-dividend yields, while PDE are considered more for American options with early-excercise.
Users can access online stock information as parameter inputs for more real-scenario analysis.


## Installation Steps
#### 1. Get this repository by running:
```
git clone https://github.com/wynonawan/OptionQuantLab.git
```


#### 2. Most libraries run in python3.9 or above. If don't have the upgraded version, you can upgrade your python by running:
```
brew install python@3.9
```
   Or, you can create a virtual python3.9 environment and run your code in it, without changing your local python package version, 
```
python3.9 -m venv venv
source venv/bin/activate
```
If you use conda for virtual environment already, then run
```
conda create -n quantenv python=3.9
conda activate quantenv
```


#### 3. Made sure your `pip` is updated:
```
pip install --upgrade pip
```


#### 4. Install all the python libraries necessary to run the codes (if you don't have so already)
```
pip install yfinance xgboost scikit-learn matplotlib pandas numpy statsmodels sympy
```

Note that `xgboost` compiled with OpenMP looks for `libomp.dylib`. If you haven't has it installed, run:
```
brew install libomp
```

## * Running the Quant Project

#### 1. Option pricing using Black-Scholes equations and Crank-Nicolson PDE

The functions are implemented in script `pricing_processor.py` as an import fo `run_pricing_analysis.py`. To simply get the call and put pricings with default parameters (stock, strike, expiry, volatility), run:
```
python run_pricing_analysis.py
```
Below are example plots for the call and put options according to BS equations

<img width="1329" height="1067" alt="call_option_3D_map" src="https://github.com/user-attachments/assets/4bd5366f-4ba9-4bad-acab-1f6a3137b155" />

<img width="1329" height="1067" alt="put_option_3D_map" src="https://github.com/user-attachments/assets/12f5bc64-1545-47b2-a44d-e408ecf61556" />



#### 2. Real-stock data
To obtain a specific company information, including the latest closing stock price, strike prices at specific dates, use command option `-c`, `-e` and `-K` run:
```
python run_pricing_analysis.py -c AAPL -e 2025-09-05 -K 140
```
If such date isn't a valid expiry date, the code would print out available dates. Similarly, if the strike price isn't available on such date, the code would print out available strike prices. Then just adjust the input in command line to rerun.

This returns the real option prices accordingly.

#### 3. Running the Greeks
The Greeks parameters are incorporated in script `greeks_processor.py`
You can also run the Greek calculations including Delta, Gamma, Theta, and Vega that measure how the option prices are sensitive to underlying stock price, time and volatility.
```
python run_pricing_analysis.py --run-greeks
```

