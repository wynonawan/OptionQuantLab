# OptionQuantLab
This quant lab repo is built for exploring and visualizing various option theories for peronsal interest, as well as evaluating strategies under real input.
It implements option pricing methods, including Black-Scholes equations, Crank-Nicolson PDE.
BS equations here are typically used for European options with non-dividend yields, while PDE are considered more for American options with early-excercise.  This project also integrates greeks parameter calculation to observe market behaviors over time, and simulates dynamic delta hedging over stock progression, which is oftenly used by market makers to neutralize risks. 

Users can access online stock information as parameter inputs for more real-scenario analysis.

## Contents

- [Install](#installation-steps)
- Running the Quant Project:
  - [Black-Scholes Equations](https://github.com/wynonawan/OptionQuantLab?tab=readme-ov-file#1-option-pricing-using-black-scholes)
  - [Crank-Nicolson PDE Option Pricing](https://github.com/wynonawan/OptionQuantLab?tab=readme-ov-file#2-crank-nicolson-pde-for-option-pricing)
  - [Input Real Stock Data](https://github.com/wynonawan/OptionQuantLab?tab=readme-ov-file#3-real-stock-data)
  - [Greeks Calculation](https://github.com/wynonawan/OptionQuantLab?tab=readme-ov-file#4-running-the-greeks)
  - [Dynamic Delta Hedging on MC Simulation](https://github.com/wynonawan/OptionQuantLab?tab=readme-ov-file#5-dynamic-delta-hedging-under-mc-stock-simulation)

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

### 1. Option pricing using Black-Scholes Equations

The **Black-Scholes formulas** for European call and put option pricing:

#### Call Option

$$
C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2),
$$

#### Put Option

$$
P = K e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1),
$$

where

$$
d_1 = \frac{\ln(S/K) + \left(r + \frac{\sigma^2}{2}\right) T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

**Variables:**

- `S` = Stock price  
- `K` = Strike price  
- `T` = Time to maturity (in years)  
- `r` = Risk-free interest rate  
- `σ` = Volatility of the underlying asset  
- `N(x)` = Cumulative distribution function (CDF) of the standard normal distribution  

The equations are implemented in script `pricing_processor.py` as an import fo `run_pricing_analysis.py`. To get the call and put pricings with default parameters (stock, strike, expiry, volatility), run:
```
python run_pricing_analysis.py
```
The input values can be changed by using command options stock `-S`, strike `-K`, interest rate `-r`, volatility `-sg`, time to maturity `-T`, dividend `-y`.

You can see the printout in your terminal / interface:

<img width="351" height="320" alt="BS-pricing-model" src="https://github.com/user-attachments/assets/618321e2-a0a0-4b88-99e8-2bee48d70d8f" />

In the meantime, figures are produced in directory `plots`. Below are example plots for the call and put options according to BS equations and default values:


<img width="1329" height="1067" alt="call_option_3D_map" src="https://github.com/user-attachments/assets/4bd5366f-4ba9-4bad-acab-1f6a3137b155" />

<img width="1329" height="1067" alt="put_option_3D_map" src="https://github.com/user-attachments/assets/12f5bc64-1545-47b2-a44d-e408ecf61556" />

### 2. Crank-Nicolson PDE for Option Pricing

The Crank-Nicolson method is a finite difference scheme used to numerically solve the **Black-Scholes Partial Differential Equation (PDE)** for European options:

#### Black-Scholes PDE

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

where:

- `V(S,t)` = option value as a function of stock price `S` and time `t`  
- `σ` = volatility of the underlying asset  
- `r` = risk-free interest rate  
- `S` = underlying asset price  

---

#### Crank-Nicolson Discretization

For a grid with stock price steps `i` and time steps `j`, the Crank-Nicolson scheme approximates the PDE as:

$$
\alpha_i V_{i-1}^{j+1} + (1 + 2\alpha_i) V_i^{j+1} + \alpha_i V_{i+1}^{j+1} = 
\alpha_i V_{i-1}^{j} + (1 - 2\alpha_i) V_i^{j} + \alpha_i V_{i+1}^{j}
$$



where:

- $\alpha_i = \frac{1}{4} \sigma^2 i^2 \Delta t - \frac{1}{4} r i \Delta t$  
- $\Delta t$ = time step
- $V_{i}^{j}$ = value at current time step
- $V_{i}^{j+1}$ = value at next time step


This results in a **tridiagonal system of equations** that can be solved iteratively to get option prices at all grid points.

---

#### Notes

- Crank-Nicolson is **implicit and stable**, and provides a second-order accurate approximation in both time and space.  
- It is widely used for **European call/put options** and can be extended to handle **American options** with early exercise constraints.
- See [reference](http://www.goddardconsulting.ca/matlab-finite-diff-crank-nicolson.html)


The call option pricing visualization from Crank-Nicolson PDE is indicated in below figure:

<img width="2295" height="2065" alt="call_option_3D_map_PDE" src="https://github.com/user-attachments/assets/7e3cf428-ba7d-4c81-9a05-7682dad5e0ee" />

You can also check the error convergence comparing the BS and PDE models. The below log plots are showing how the error converges as more points are added.

<img width="1731" height="1362" alt="pde_error_convergence" src="https://github.com/user-attachments/assets/afb7e757-ae19-49df-a738-d06be86bc2d1" />



### 3. Real-stock Data
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



### 4. Running the Greeks
The Greeks parameters are incorporated in script `greeks_processor.py`
You can also run the Greek calculations including Delta, Gamma, Theta, and Vega that measure how the option prices are sensitive to underlying stock price, time and volatility.

**Delta: Stock price sensitive**



$$
\Delta_{\text{call}} = \frac{\partial C}{\partial S} = N(d_1)
$$



$$
\Delta_{\text{put}} = \frac{\partial P}{\partial S} = N(d_1) - 1
$$

**Gamma: Delta to stock price sensitive**

$$
\Gamma = \frac{\partial^{2} C}{\partial S^{2}} = \frac{\partial^{2} P}{\partial S^{2}} = \frac{\partial^{2} P}{\partial S^{2}}
$$

**Vega: Volatility sensitive**

$$
\text{Vega} = \frac{\partial P}{\partial \sigma} = \frac{\partial P}{\partial \sigma} = S \, N'(d_1) \sqrt{T}
$$

**Theta: time sensitive**



$$
\Theta_{\text{call}} = \frac{\partial C}{\partial T}  = -\frac{S N'(d_1) \sigma}{2 \sqrt{T}} - r K e^{-r T} N(d_2)
$$



$$
\Theta_{\text{put}} = \frac{\partial P}{\partial T} = -\frac{S N'(d_1) \sigma}{2 \sqrt{T}} + r K e^{-r T} N(-d_2)
$$

You can calculate all these greek values by adding `--run-greeks` command option as,

```
python run_pricing_analysis.py --run-greeks
```

With default values, or if you specify a company `-c`, this code will return a table with all the greek values.
Below is an example for the default values

<img width="284" height="204" alt="call_greeks" src="https://github.com/user-attachments/assets/0288b434-1da9-4072-8745-3914dd3b0bea" />

<img width="264" height="203" alt="put_greeks" src="https://github.com/user-attachments/assets/a8b6b170-2552-4a4f-88d8-8e103a9bbd0a" />


In addition, this command also plots the progression of greek values in respect to time to maturity. 
Below four example figures show call option greek values according to time to maturity. Delta is in between 0 and 1, Theta is always negative, while Vega is always positive.
<img width="2100" height="1463" alt="Delta_vs_T" src="https://github.com/user-attachments/assets/4f6ee159-e369-4c12-9c20-a3938596889c" />
<img width="2127" height="1463" alt="Gamma_vs_T" src="https://github.com/user-attachments/assets/53944421-fffe-4497-8cba-0b0f388d7e48" />
<img width="2060" height="1463" alt="Theta_vs_T" src="https://github.com/user-attachments/assets/cc8cf175-b0be-4484-b076-fe61e061f9ad" />
<img width="2060" height="1463" alt="Vega_vs_T" src="https://github.com/user-attachments/assets/2b330e2f-6e5c-4652-b644-7e57d036264c" />



### 5. Dynamic Delta Hedging under MC Stock Simulation

For neutralizing the price risk of an option position with respect to small changes in the underlying asset price, the method of Delta hedging is commonly used. It builts a portfolio that contains options and the stock shares, and offsets the option's price sensitivity to the underlying aseet by taking a position in the underlying asset. 
For instance:

Hedged portfolio:

$$
\Pi = V - \Delta S
$$

where $\Pi$ is the value of the hedged portfolio.

Shares to hold per option:

$$
\text{Shares} = \Delta \times (\text{number of options})
$$

Example for call/put options:

$$
\Delta_{\text{call}} = N(d_1), \quad \Delta_{\text{put}} = N(d_1) - 1
$$

To maintain a neutral-risk position:

$$
d\Pi \approx dV - \Delta dS \rightarrow \text{should be approximately zero}
$$

For Ddynamic (Discrete-time) hedging:

Since delta would change over time, you can re-hedge periodically by shorting or longing shares to maintain the best stock position
At time \(t_j\), rehedge using the current delta:  

$$
\text{Stock position at } t_j = \Delta_j \times (\text{number of options})
$$

In order to test out hedging strategy, the code also performs MC simulation over stock price following Geometric Brownian motion (GBM):

$$
S_{t+\Delta t} = S_t \times \exp \Big( (r - \frac{1}{2}\sigma^2) \Delta t + \sigma \sqrt{\Delta t} Z \Big)
$$

where:  

- $\Delta t$ = time step  
- $S_t$ = current stock price  
- $S_{t+\Delta t}$ = stock price at next time step
- $Z \sim N(0,1)$ = standard normal random variable


Running command option `--run-greeks`, you will automatically get mc simulation of stock price and delta hedging calculations / plots. Currently the strategy is only for longing call option. You can input the number of mc paths you will want to see `-M`, the number of time steps you want to set `-N`. To edit the number of options, you can use `--options`.
```
python run_pricing_analysis.py --run-greeks -M 20 -N 20 -options 200
```

##### Hedging example: Long call options
The below print out returns 10 mc paths with estimated P&L using dynamic delta hedging while longing call options.
You can see that the P&L off all paths are approximately zero, which is what we expect to see.

<img width="300" height="174" alt="dynamic_hedging" src="https://github.com/user-attachments/assets/443b47b7-6e27-49bc-87e5-64cd27575bdb" />

Below figure is an example of the dynamic hedging visualization with default values and 100 options. At each time step, it returns strategy as to how many extra stocks should be shorted / longed, to result in final P&L of almost zero.
<img width="2260" height="1467" alt="delta_hedge_simulation_3" src="https://github.com/user-attachments/assets/f76b49a4-d648-46c8-9ae4-2c9fc4d77180" />

