# OptionQuantLab
This quant lab repo is built for exploring and visualizing various financial theories, as well as observing how they behave with applying real market parameters.
It implements multiple calculations for options, including Black-Scholes equations, Crank-Nicolson PDE and Greeks.
BS equations here are typically used for European options with non-dividend yields, while PDE are considered more for American options with early-excercise.
Users can access online stock information as parameter inputs for more real-scenario analysis.


## Installation Steps
### 1. Get this repository by running:
```
git clone https://github.com/wynonawan/OptionQuantLab.git
```


### 2. Most libraries run in python3.9 or above. If don't have the upgraded version, you can upgrade your python by running:
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


### 3. Made sure your `pip` is updated:
```
pip install --upgrade pip
```


### 4. Install all the python libraries necessary to run the codes (if you don't have so already)
```
pip install yfinance xgboost scikit-learn matplotlib pandas numpy statsmodels sympy tabulate termcolor
```

Note that `xgboost` compiled with OpenMP looks for `libomp.dylib`. If you haven't has it installed, run:
```
brew install libomp
```

## * Running the Quant Project

### 1. Option pricing using Black-Scholes equations and Crank-Nicolson PDE

The functions are implemented in script `pricing_processor.py` as an import fo `run_pricing_analysis.py`. To simply get the call and put pricings with default parameters (stock, strike, expiry, volatility), run:
```
python run_pricing_analysis.py
```
The input values can be changed by using command options stock `-S`, strike `-K`, interest rate `-r`, volatility `-sg`, time to maturity `-T`, dividend `-y`.

You can see the printout in your terminal / interface:

<img width="351" height="320" alt="BS-pricing-model" src="https://github.com/user-attachments/assets/618321e2-a0a0-4b88-99e8-2bee48d70d8f" />

In the meantime, figures are produced in directory `plots`. Below are example plots for the call and put options according to BS equations and default values:


<img width="1329" height="1067" alt="call_option_3D_map" src="https://github.com/user-attachments/assets/4bd5366f-4ba9-4bad-acab-1f6a3137b155" />

<img width="1329" height="1067" alt="put_option_3D_map" src="https://github.com/user-attachments/assets/12f5bc64-1545-47b2-a44d-e408ecf61556" />


The call option pricing visualization from Crank-Nicolson PDE is indicated in below figure:

<img width="2295" height="2065" alt="call_option_3D_map_PDE" src="https://github.com/user-attachments/assets/7e3cf428-ba7d-4c81-9a05-7682dad5e0ee" />

You can also check the error convergence comparing the BS and PDE models. The below log plots are showing how the error converges as more points are added.

<img width="1731" height="1362" alt="pde_error_convergence" src="https://github.com/user-attachments/assets/afb7e757-ae19-49df-a738-d06be86bc2d1" />



### 2. Real-stock data
To obtain a specific company information, including the latest closing stock price, strike prices at specific dates, use command option `-c`, `-e` and `-K` run:
```
python run_pricing_analysis.py -c AAPL -e 2025-09-05 -K 140
```
If such date isn't a valid expiry date, the code would print out available dates. Similarly, if the strike price isn't available on such date, the code would print out available strike prices. Then just adjust the input in command line to rerun.

If both strike price and the date are available, you obtain a table with the option information. The same inputs are then also used for BS option calculation for comparison, which automatically updates the previous table.
<img width="421" height="324" alt="Company-stock-info" src="https://github.com/user-attachments/assets/f88fcd3d-4d0f-4d06-8a48-c534ce89a233" />

This prints out the stock information and returns the real option prices accordingly on a table.

It also inputs the parameter values from extracted data online to the Black-Scholes model and produce similar plots:
<img width="2329" height="2067" alt="call_option_3D_map" src="https://github.com/user-attachments/assets/03fd21b3-54aa-4e4a-9d34-a53ef25b1c09" />
<img width="2329" height="2067" alt="put_option_3D_map" src="https://github.com/user-attachments/assets/c4ee8672-6795-4a02-8bc0-191fb25f2e93" />



### 3. Running the Greeks
The Greeks parameters are incorporated in script `greeks_processor.py`
You can also run the Greek calculations including Delta, Gamma, Theta, and Vega that measure how the option prices are sensitive to underlying stock price, time and volatility.
```
python run_pricing_analysis.py --run-greeks
```
With default values, or if you specify a company `-c`, this code will return a table.
Below is an example for the default values

<img width="284" height="204" alt="call_greeks" src="https://github.com/user-attachments/assets/0288b434-1da9-4072-8745-3914dd3b0bea" />

<img width="264" height="203" alt="put_greeks" src="https://github.com/user-attachments/assets/a8b6b170-2552-4a4f-88d8-8e103a9bbd0a" />



Below four figures show call option greek values according to time to Maturity. Delta is in between 0 and 1, Theta is always negative, while Vega is always positive.
<img width="2100" height="1463" alt="Delta_vs_T" src="https://github.com/user-attachments/assets/4f6ee159-e369-4c12-9c20-a3938596889c" />
<img width="2127" height="1463" alt="Gamma_vs_T" src="https://github.com/user-attachments/assets/53944421-fffe-4497-8cba-0b0f388d7e48" />
<img width="2060" height="1463" alt="Theta_vs_T" src="https://github.com/user-attachments/assets/cc8cf175-b0be-4484-b076-fe61e061f9ad" />
<img width="2060" height="1463" alt="Vega_vs_T" src="https://github.com/user-attachments/assets/2b330e2f-6e5c-4652-b644-7e57d036264c" />



### 4. Dynamic Delta Hedging under MC stock simulation
Under the same command option `--run-greeks`, it automatically runs a MC simulation of stock progression using formula of geometric brownian motion. You can input the number of mc paths you will want to see `-M`, the number of time steps you want to set `-N`.
In the meantime, it produces a strategy for dynamic hedging by longing call options. You can also edit the number of call options `--options`. The code plots the first five paths.
```
python run_pricing_analysis.py --run-greeks -M 20 -N 20 -options 200
```
Below figure is an example of the dynamic hedging visualization with default values and 100 options. At each time step, it returns strategy as to how many extra stocks should be shorted / longed.
<img width="2260" height="1467" alt="delta_hedge_simulation_3" src="https://github.com/user-attachments/assets/f76b49a4-d648-46c8-9ae4-2c9fc4d77180" />

