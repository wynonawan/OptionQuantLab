'''
	This script uses 
        1. Black Scholes equations to solve for both call and put option prices (European Option)
        2. Crank-Nicolson PDE to solve for call prices (American option early excercise)
        3. Plots 3D and 2D maps

	This is used as an import script in `run_pricing_analysis.py`
'''

import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import sqrt



# Returns call or put option prices using Black-Scholes equations
def Option_price(S, K, T, r, sigma, y=0, option='call'):

    if T <= 0:
        if option == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r - y + (sigma**2)/2 ) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    estimated_call = S * np.exp(-y * T) * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
    estimated_put  = K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(-y * T) * st.norm.cdf(-d1)

    if option == 'call':
        return float(estimated_call)
    else:
        return float(estimated_put)



# Crank-Nicolson PDE for European call
def crank_nicolson_call(S0, K, T, r, sigma, Smax_factor=3.0, M=200, N=200):

    Smax = Smax_factor * K + S0
    dS = Smax / M
    dt = T / N
    S = np.linspace(0.0, Smax, M+1)        # 0..M

    # payoff at maturity
    V = np.maximum(S - K, 0.0)
    # interior indices
    idx = np.arange(1, M)                  # 1..M-1
    m = len(idx)                           # = M-1
    S_i = S[idx]
    sigma_2 = sigma * sigma

    # Precompute discrete operator L coefficients:
    # L V_i = a_i * V_{i-1} + b_i * V_i + c_i * V_{i+1}
    a = 0.5 * sigma_2 * (S_i**2) / (dS**2) - 0.5 * r * S_i / dS
    b = - sigma_2 * (S_i**2) / (dS**2) - r
    c = 0.5 * sigma_2 * (S_i**2) / (dS**2) + 0.5 * r * S_i / dS

    # CN matrices: Left = I - 0.5*dt*L, Right = I + 0.5*dt*L
    L_lower = -0.5 * dt * a[1:]              # length m-1 (for lower diag of Left)
    L_diag  = 1.0 - 0.5 * dt * b             # length m (main diag of Left)
    L_upper = -0.5 * dt * c[:-1]             # length m-1 (upper diag of Left)

    R_lower =  0.5 * dt * a[1:]              # length m-1 (lower diag of Right)
    R_diag  =  1.0 + 0.5 * dt * b            # length m (main diag of Right)
    R_upper =  0.5 * dt * c[:-1]             # length m-1 (upper diag of Right)

    # Build sparse matrices (m x m)
    A = diags([L_lower, L_diag, L_upper], offsets=[-1, 0, 1], shape=(m, m), format='csc')
    B = diags([R_lower, R_diag, R_upper], offsets=[-1, 0, 1], shape=(m, m), format='csc')

    # time-stepping backward from maturity to 0
    for n in range(N):
        t = T - n * dt            # current time (backwards)
        V_old = V.copy()
        rhs = B.dot(V_old[1:-1])  # interior RHS

        # Boundary contributions: V_0 = 0 for call; V_M = Smax - K*exp(-r * tau)
        # Use the boundary values at the current time level (V_old)
        V_0 = 0.0
        V_M = Smax - K * np.exp(-r * (t - dt))  # approximate boundary at next step time

        # add boundary terms coming from R matrix (explicit side)
        # lower boundary contributes to rhs[0] via R_lower[0] * V_0
        # upper boundary contributes to rhs[-1] via R_upper[-1] * V_M
        if m > 0:
            rhs[0] += R_lower[0] * V_0
            rhs[-1] += R_upper[-1] * V_M

        # solve left system for new interior values
        V_interior = spsolve(A, rhs)

        # update full grid
        V[1:-1] = V_interior
        V[0] = V_0
        V[-1] = V_M

    return float(np.interp(S0, S, V))

def pde_price_wrapper(S0, K, T, r, sigma, M,N,Smax_factor=4.0):
    return crank_nicolson_call(S0, K, T, r, sigma, Smax_factor=Smax_factor, M=M, N=N)

# add features to plotting
def set_ticks(axis, values, max_ticks=10):
    n = len(values)
    step = max(1, n // max_ticks)
    ticks_pos = values[::step]
    axis.set_ticks(ticks_pos)
    axis.set_ticklabels([f"{v:.1f}" for v in ticks_pos])


def add_arrows(ax, x, y, z, invert_x=False):
    # Arrow length as 10% of axis ranges
    len_x = (x[1] - x[0]) * 0.1
    len_y = (y[1] - y[0]) * 0.1
    len_z = (z[1] - z[0]) * 0.1
    x0, y0, z0 = x[0], y[0], z[0]

    x_dir = -len_x if invert_x else len_x

    ax.quiver(x0, y0, z0, x_dir, 0, 0, color='green', arrow_length_ratio=0.2, linewidth=1.8)
    ax.quiver(x0, y0, z0, 0, len_y, 0, color='blue', arrow_length_ratio=0.2, linewidth=1.8)
    ax.quiver(x0, y0, z0, 0, 0, len_z, color='red', arrow_length_ratio=0.2, linewidth=1.8)

    ax.text(x0 + x_dir * 1.2, y0, z0, 'S', color='green', fontsize=11)
    ax.text(x0, y0 + len_y * 1.2, z0, 'T', color='blue', fontsize=11)
    ax.text(x0, y0, z0 + len_z * 1.5, 'Price', color='red', fontsize=11)


def run_pricing(S, K, T, r, sigma, y, output):

    estimated_call = Option_price(S, K, T, r, sigma, y, option='call')
    estimated_put  = Option_price(S, K, T, r, sigma, y, option='put')

    Pricing_table = pd.DataFrame({
        "Parameter": [
            "Stock Price", "Strike Price", "Time to Maturity",
            "Interest Rate", "Volatility", "Dividend Yield", "BS Estimated Call Price", "BS Estimated Put Price"
        ],
        "Value": [S, K, T, r, sigma, y, estimated_call, estimated_put]
    })

    Pricing_table["Value"] = Pricing_table["Value"].round(4)
    print('\n\n')
    print("Option Pricing with BS estimation (Non-Dividend)")
    print(Pricing_table.to_string(index=False))
    print('\n\n')

    # Plotting code assited by AI
    S_range, T_range  = np.linspace(0.5*K, 1.5*K, 100), np.linspace(0, 1.0, 100)
    S_bin, T_bin= np.meshgrid(S_range, T_range)

    option_price = np.vectorize(Option_price)

    call_prices = option_price(S_bin, K, T_bin, r, sigma, option='call', y=0)
    put_prices = option_price(S_bin, K, T_bin, r, sigma, option='put', y=0)

    BS_label = f"Stike Price:{K}, Interest Rate:{r}, Volatility:{sigma}, Dividend:{y}"


    figure1 = plt.figure(figsize=(10, 7))
    figure_call = figure1.add_subplot(111, projection='3d')
    subplot_call = figure_call.plot_surface(S_bin, T_bin, call_prices, cmap='cividis', edgecolor='none')


    figure_call.set_title(f'Call Option Price 3D Mapping\n({BS_label})')
    figure_call.set_xlabel('Stock Price')
    figure_call.set_ylabel('Time to Maturity')
    figure_call.set_zlabel('Option Price')

    figure_call.invert_xaxis()
    add_arrows(figure_call, figure_call.get_xlim(), figure_call.get_ylim(), figure_call.get_zlim(), invert_x=True)

    figure1.colorbar(subplot_call, ax=figure_call, shrink=0.5, aspect=10, label='Price')
    plt.tight_layout()
    figure1.savefig(os.path.join(output,"call_option_3D_map.png"), dpi=300, bbox_inches='tight')
    plt.close(figure1)
    print(f"call_option_3D_map.png saved in \{output}")

    figure2 = plt.figure(figsize=(10, 7))
    figure_put = figure2.add_subplot(111, projection='3d')
    subplot_put = figure_put.plot_surface(S_bin, T_bin, put_prices, cmap='plasma', edgecolor='none')
    figure_put.set_title(f'Put Option Price 3D Map \n({BS_label})')
    figure_put.set_xlabel('Stock Price')
    figure_put.set_ylabel('Time to Maturity T')
    figure_put.set_zlabel('Option Price')
    add_arrows(figure_put, figure_put.get_xlim(), figure_put.get_ylim(), figure_put.get_zlim(), invert_x=False)
    figure2.colorbar(subplot_put, ax=figure_put, shrink=0.5, aspect=10, label='Price')
    plt.tight_layout()
    figure2.savefig(os.path.join(output,"put_option_3D_map.png"), dpi=300, bbox_inches='tight')
    plt.close(figure2)
    print(f"put_option_3D_map.png saved in \{output}")

    # Call 2D Map
    plt.figure(figsize=(8, 6))
    plt.imshow(call_prices, origin='lower', aspect='auto', cmap='viridis',
           extent=[S_range[0], S_range[-1], T_range[0], T_range[-1]])
    plt.colorbar(label='Call Price')
    plt.title('Call Option Price 2D Heatmap')
    plt.xlabel('Stock Price')
    plt.ylabel('Time to Maturity')

    ax = plt.gca()
    set_ticks(ax.xaxis, S_range)
    set_ticks(ax.yaxis, T_range)

    plt.tight_layout()
    plt.savefig(os.path.join(output,"call_option_2D_map.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"call_option_2D_map.png saved in \{output}")

    # Put 2D Map
    plt.figure(figsize=(8, 6))
    plt.imshow(put_prices, origin='lower', aspect='auto', cmap='plasma',
           extent=[S_range[0], S_range[-1], T_range[0], T_range[-1]])
    plt.colorbar(label='Put Price')
    plt.title('Put Option Price 2D Heatmap')
    plt.xlabel('Stock Price')
    plt.ylabel('Time to Maturity')

    ax = plt.gca()
    set_ticks(ax.xaxis, S_range)
    set_ticks(ax.yaxis, T_range)

    plt.tight_layout()
    plt.savefig(os.path.join(output, "put_option_2D_map.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"put_option_2D_map.png saved in \{output}")


    ##############  BS PDE Pricing ##############

    pde_price = crank_nicolson_call(S,K,T,r,sigma,Smax_factor=4.0,M=400,N=400)
    print("CN PDE price:", pde_price)


    Ms = [50,100,200,400,800]
    errs = []
    for M in Ms:
        N = M  # keep dt ~ dS
        p = pde_price_wrapper(S, K, T, r, sigma, M,N, Smax_factor=4.0)
        errs.append(abs(estimated_call - p))
        print("M,N=",M,": PDE=",p,", abs err=",errs[-1])

    os.makedirs(output, exist_ok=True)

    plt.loglog(Ms, errs, marker='o')
    plt.xlabel('M (spatial grid points)')
    plt.ylabel('Absolute error')
    plt.title('Grid convergence (CN)')
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(output,"pde_price_surface.png"), dpi=300, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    run_pricing()
