
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


def implied_volatility(mkt_price, S, K, T, r, option='call', d=0.0, sg0=0.2):
    f = lambda sg: Option_price(S,K,T,r,sg,option,d) - mkt_price
    # bracket or use a robust solver
    try:
        res = opt.brentq(f, 1e-8, 5.0, maxiter=200)
        return res
    except Exception:
        # fallback to newton using vega
        try:
            sg = sg0
            for i in range(50):
                price = Option_price(S,K,T,r,sg,option,d)
                vega = bs_vega(S,K,T,r,sg,d)
                if vega == 0:
                    break
                sg -= (price - mkt_price) / vega

                sg = max(1e-8, sg)
            return sg
        except Exception:
            return np.nan


def run_Greeks(S, K, T, r, sigma, y, output):

    d1, d2 = get_d1d2(S, K, T, r, sigma, y=0, option='call')
    call_Delta  =  get_Delta(d1, y, T, option='call')
    call_Theta  =  get_Theta(S, d1, d2, K, sigma, y, T, r, option='call')


    d1, d2 = get_d1d2(S, K, T, r, sigma, y=0, option='put')
    put_Delta  =  get_Delta(d1, y, T, option='put')
    put_Theta  =  get_Theta(S, d1, d2, K, sigma, y, T, r, option='put')

    option_Gamma = get_Gamma(S, d1, sigma, y, T)
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
    print("Call Option Greeks Parameter Estimate")
    print(Greeks_call_table.to_string(index=False))
    print('\n\n')

    print('\n\n')
    print("Put Option Greeks Parameter Estimate")
    print(Greeks_put_table.to_string(index=False))
    print('\n\n')


    Delta_vals = []
    Theta_vals = []
    Gamma_vals = []
    Vega_vals = []

    T_range = np.linspace(0.01, 1, 100)
    for T in T_range:
        d1_vals, d2_vals = get_d1d2(S, K, T, r, sigma, y=0, option='call')

        Delta_vals.append(get_Delta(d1_vals, y, T, option='put'))
        Theta_vals.append(get_Theta(S, d1_vals, d2_vals, K, sigma, y, T, r, option='put'))
        Gamma_vals.append(get_Gamma(S, d1_vals, sigma, y, T))
        Vega_vals.append(get_Vega(S, d1_vals, y, r, T))

    plt.figure(figsize=(10,6))
    plt.plot(T_range, Delta_vals, label='Delta')
    plt.plot(T_range, Gamma_vals, label='Gamma')
    plt.plot(T_range, Theta_vals, label='Theta')
    plt.plot(T_range, Vega_vals,  label='Vega')

    plt.xlabel('Time to Maturity')
    plt.ylabel('Greek Value')
    plt.title('Greeks vs Time to Maturity')
    plt.legend()
    plt.grid(True)
#    plt.show()

    plt.savefig(os.path.join(output, "call_Greeks_vs_T.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"call_Greeks_vs_T.png saved in \{output}")


if __name__ == "__main__":
    run_Greeks()
