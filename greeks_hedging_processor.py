```
       This script is used as an import to `run_pricing_processor.py`
       It implements these functions:
       1. Greeks calculation of options given stock price, strike price, volatility, interest, dividend and time to maturity. 
       2. MC simulation of stock with geometric brownian motion formula
       3. Performs dynamic delta hedging for the simulated path and returns hedging position at each time step. Givens P&L of approximately zero each time.


```


from tabulate import tabulate

import numpy as np
import scipy.stats as st
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_d1d2(S, K, T, r, sigma, y=0, option='call'):

    d1 = (np.log(S / K) + (r - y + (sigma**2)/2 ) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def get_Delta(d1, y, T, option='call'):

    Delta_call = np.exp(-y * T) * st.norm.cdf(d1)
    Delta_put  = - np.exp(-y * T) * ( 1 - st.norm.cdf(d1))

    if option == 'call':
        return Delta_call
    if option == 'put':
        return Delta_put


def get_Gamma(S0, d1, sigma, y, T):

    Gamma = ( st.norm.pdf(d1) * np.exp(-y * T) ) / (S0 * sigma * np.sqrt(T))
    return Gamma


def get_Theta(S0, d1, d2, K, sigma, y, T, r, option='call'):

    Theta_call = - (S0 * st.norm.pdf(d1) * sigma * np.exp(-y * T)) / (2 * np.sqrt(T)) + y * S0 * np.exp(-y * T) * st.norm.cdf(d1) - r * K * np.exp(-r * T) * st.norm.cdf(d2)
    Theta_put  = - (S0 * st.norm.pdf(d1) * sigma * np.exp(-y * T)) / (2 * np.sqrt(T)) - y * S0 * np.exp(-y * T) * st.norm.cdf(-d1)+ r * K * np.exp(-r * T) * st.norm.cdf(-d2)

    if option == 'call':
        return - Theta_call
    if option == 'put':
        return Theta_put


def get_Vega(S0, d1, y, r, T):
    Vega = S0 * np.exp(-y * T) * np.sqrt(T) * st.norm.pdf(d1)
    return Vega

# This function generate multiple MC paths for stock price progression
# Delta hedging strategy is applied on longing call option
def Delta_hedge_simulate(S0, K, T, r, sigma, y, d1, N, M, option_count, call_price, output):

    np.random.seed(42)
    t_steps = np.linspace(0, T, N)
    dt = T/N       # 0,dt,2dt...T
    paths = np.zeros((M, N))
    paths[:,0] = S0

    print('\n\n')
    print("Dynamic Hedging on MC stock simulation:")

    for m in range(M):

        d1_init, _ = get_d1d2(S0, K, T, r, sigma, y=0, option='call')
        Delta_prev = get_Delta(d1_init, y, T, option='call')

        shares = - option_count * Delta_prev   # long option position
        cash_init = - shares * S0              # initiate cash flow
        cash = cash_init
        call_value = call_price

        if m < 5: # get plots points for 5 paths
            stock_prices_path, delta_path, shares_path, time_steps = [], [], [], []
            stock_prices_path.append(S0)
            delta_path.append(Delta_prev)
            shares_path.append(-shares)
            time_steps.append(0)

        for n in range(1, N):   # Dynamic Delta hedging on MC simulation

            Z = np.random.normal()
            paths[m,n] = paths[m, n-1] * np.exp((r - sigma**2/2) * dt + sigma * np.sqrt(dt) * Z ) #stock MC, GBM

            d1_hedge, _ = get_d1d2(paths[m,n], K, T-n*dt, r, sigma, y=0, option='call')
            Delta_now = get_Delta(d1_hedge, y, T-n*dt, option='call')  # dc/dS
            Delta_change = Delta_now - Delta_prev

            stock_change = paths[m,n] - paths[m,n-1]

            cash += Delta_change * paths[m,n] * option_count
            shares_change = Delta_change * option_count  # short shares
            shares -= shares_change                      # hedging position
            call_value += ( stock_change * Delta_prev)   # dc=dS*(dc/dS)

            Delta_prev = Delta_now

            if m < 5:  # plot 5 paths
                stock_prices_path.append(paths[m, n])
                delta_path.append(Delta_now)
                shares_path.append(shares_change)
                time_steps.append(n*dt)

        PNL = shares * paths[m,-1] + cash + call_value * option_count - call_price * option_count
        print(f"MC Path {m} P&L:", PNL)  # prints P&L for all paths

        # plot 5 paths
        if m < 5:

            norm = plt.Normalize(min(delta_path), max(delta_path))
            cmap = plt.cm.coolwarm                   # blue for low delta, red for high delta
            colors = cmap(norm(delta_path))

            plt.figure(figsize=(8, 5))
            plt.plot(time_steps, stock_prices_path, color='gray', linewidth=1.2, zorder=1)
            sc= plt.scatter(time_steps, stock_prices_path, c=delta_path, cmap=cmap, edgecolor='black', s=20, zorder=2)
            plt.xlabel("Time Steps")
            plt.ylabel("Stock Price")
            plt.title(f"MC Path {m}: Dynamic Delta Hedging \n{option_count} Options, P&L: {PNL}")

            for i in range(N):
                label = f"Δ={delta_path[i]:.2f}\nshort:{shares_path[i]:.1f}"
                plt.annotate(
                    label,
                    (time_steps[i], stock_prices_path[i]),
                    textcoords="offset points",
                    xytext=(6, 2),
                    ha='center',
                    fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
                )

            cbar = plt.colorbar(sc)
            cbar.set_label("Delta Value")

            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            save_path = os.path.join(output, f"delta_hedge_simulation_{m}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()



# This function runs the greek estimation functions
# It also implements Delta hedging for longing a call option
def run_greeks_hedging(S, K, T, r, sigma, y, output, call_price, N, M, options):

    # calculated all greeks

    d1, d2 = get_d1d2(S, K, T, r, sigma, y=0, option='call')
    call_Delta  =  get_Delta(d1, y, T, option='call')
    call_Theta  =  get_Theta(S, d1, d2, K, sigma, y, T, r, option='call')


    d1, d2 = get_d1d2(S, K, T, r, sigma, y=0, option='put')
    put_Delta  =  get_Delta(d1, y, T, option='put')
    put_Theta  =  get_Theta(S, d1, d2, K, sigma, y, T, r, option='put')

    option_Gamma = get_Gamma(S, d1, sigma, y, T)    # both call and put
    option_Vega  = get_Vega(S, d1, y, r, T)

    Greeks_call_table = pd.DataFrame({
        "Greek": [
            "Delta","Gamma", 
            "Theta",  "Vega"
        ],
        "Value": [call_Delta, option_Gamma, call_Theta, option_Vega]
    })

    Greeks_put_table = pd.DataFrame({
        "Greek": [
            "Delta","Gamma",          
            "Theta",  "Vega"
        ],
        "Value": [put_Delta, option_Gamma, put_Theta, option_Vega]
    })

    Greeks_call_table["Value"] = Greeks_call_table["Value"].round(4)
    Greeks_put_table["Value"] = Greeks_put_table["Value"].round(4)

    print('\n\n')
    print('\nCall Option Greeks Parameter Estimate')
    print(tabulate(Greeks_call_table, headers="keys", tablefmt="fancy_grid", showindex=False))
    print('\n\n')
    print('\nPut Option Greeks Parameter Estimate')
    print(tabulate(Greeks_put_table, headers="keys", tablefmt="fancy_grid", showindex=False))
    print('\n\n')


    # generate greek values vs. time to maturity plots

    T_range = np.linspace(0.01, 1, 100)
    BS_label = f"Stike Price:{K}, Interest Rate:{r}, Volatility:{sigma}, Dividend:{y}"

    Delta_vals, Gamma_vals, Theta_vals, Vega_vals = [], [], [], []

    for T in T_range:
        d1_vals, d2_vals = get_d1d2(S, K, T, r, sigma, y=0, option='call')

        Delta_vals.append(get_Delta(d1_vals, y, T, option='call'))
        Theta_vals.append(get_Theta(S, d1_vals, d2_vals, K, sigma, y, T, r, option='call'))
        Gamma_vals.append(get_Gamma(S, d1_vals, sigma, y, T))
        Vega_vals.append(get_Vega(S, d1_vals, y, r, T))

    greeks_input = {
        "Delta": Delta_vals,
        "Gamma": Gamma_vals,
        "Theta": Theta_vals,
        "Vega": Vega_vals
    }

    for name, values in greeks_input.items():
        plt.figure(figsize=(8, 5))
        if name == "Delta":
            plt.plot(T_range, values, label=name, color='blue')
        if name == "Gamma":
            plt.plot(T_range, values, label=name, color='green')
        if name == "Theta":
            plt.plot(T_range, values, label=name, color='orange')
        if name == "Vega":
            plt.plot(T_range, values, label=name, color='red')
        plt.xlabel('Time to Maturity')
        plt.ylabel(f'{name} Value')
        plt.title(f'{name} vs Time to Maturity \n({BS_label})')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(output, f"{name}_vs_T.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{name}_vs_T.png saved in {output}")


    # Delta hedging process using MC simulation

    Delta_hedge_simulate(S0=S, K=K, T=T, r=r, sigma=sigma, y=y, d1=d1, N=N, M=M, option_count=options, call_price=call_price, output=output)


if __name__ == "__main__":
    run_greeks_hedging()
