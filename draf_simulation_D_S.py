# draf_simulation.py
import math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,4)

# ---------------------------
# CONFIGURATION / PARAMETERS
# ---------------------------
data_centers = ["DC-A","DC-B","DC-C"]
domains = ["C","P","E","H"]  # Cyber, Power, Environment, Human
time_steps = 12  # number of simulation steps

# initial params: (P, I, D) per DC/domain
init_params = {
    "DC-A": {"C":(0.30,0.70,0.80),"P":(0.20,0.60,0.85),"E":(0.10,0.50,0.90),"H":(0.25,0.40,0.75)},
    "DC-B": {"C":(0.50,0.80,0.60),"P":(0.35,0.75,0.70),"E":(0.15,0.55,0.85),"H":(0.40,0.50,0.60)},
    "DC-C": {"C":(0.20,0.60,0.90),"P":(0.25,0.65,0.80),"E":(0.05,0.30,0.95),"H":(0.15,0.30,0.85)}
}

# domain weights (sum to 1)
weights = {"C":0.4, "P":0.3, "E":0.15, "H":0.15}

# learning/adaptation rates
alpha = 0.7  # for probability P
beta  = 0.85 # for impact I
gamma = 0.9  # for detection D

# ---------------------------
# OBSERVATION / SCENARIOS
# ---------------------------
def observed_likelihood(dc, domain, t):
    """
    Returns Obs in [0,1] for given DC, domain and time t.
    Customize to model scenario events (attack bursts, power faults, environment spike).
    """
    base = 0.0
    # Example scenarios:
    if dc=="DC-B" and domain=="C" and 3<=t<=6:  # cyber attack burst in DC-B
        base = 0.7
    if dc=="DC-A" and domain=="P" and 7<=t<=9:  # power issues in DC-A later
        base = 0.6
    if dc=="DC-C" and domain=="E" and t==5:    # environmental event
        base = 0.8
    if domain=="H":
        base = 0.08 * (1 + 0.5 * math.sin(0.6*t))  # background human error fluctuation

    # small noise
    noise = random.gauss(0, 0.05)
    val = max(0.0, min(1.0, base + 0.03*math.sin(t/2.0) + noise))
    return val

# ---------------------------
# SIMULATION
# ---------------------------
rows = []
# current state init
current = {}
for dc in data_centers:
    current[dc] = {}
    for d in domains:
        P0,I0,D0 = init_params[dc][d]
        current[dc][d] = {"P":P0, "I":I0, "D":D0}

for t in range(time_steps):
    for dc in data_centers:
        R_i_vals = {}
        for d in domains:
            P_prior = current[dc][d]["P"]
            I_prior = current[dc][d]["I"]
            D_prior = current[dc][d]["D"]

            Obs = observed_likelihood(dc, d, t)

            # update rules (exponential smoothing / linear update)
            P_new = alpha * P_prior + (1 - alpha) * Obs

            delta_I = min(1.0, 0.12 * Obs)  # impact may increase proportionally to observed stress
            I_new = min(1.0, beta * I_prior + (1 - beta) * delta_I)

            # detection may degrade slightly when Obs high, or improve slowly otherwise
            delta_D = max(-0.2, min(0.05, -0.05 * Obs))
            D_new = max(0.01, min(0.999, gamma * D_prior + (1 - gamma) * delta_D))

            R_i = (P_new * I_new) / D_new

            # write back
            current[dc][d]["P"] = P_new
            current[dc][d]["I"] = I_new
            current[dc][d]["D"] = D_new
            R_i_vals[d] = R_i

            rows.append({"time": t, "DC": dc, "domain": d, "P": P_new, "I": I_new, "D": D_new, "R_i": R_i})

        # aggregate
        R_T = sum(weights[d]*R_i_vals[d] for d in domains)
        rows.append({"time": t, "DC": dc, "domain": "TOTAL", "P": None, "I": None, "D": None, "R_i": R_T})

# results
df = pd.DataFrame(rows)
df.to_csv("draf_simulation_results.csv", index=False)

# ---------------------------
# PLOTS
# ---------------------------
# 1) Aggregated risk R_T for each DC
plt.figure()
for dc in data_centers:
    s = df[(df.DC==dc) & (df.domain=="TOTAL")]
    plt.plot(s.time, s.R_i, label=dc)
plt.xlabel("Time")
plt.ylabel("Aggregated Risk R_T")
plt.title("DRAF Simulation: Aggregated Risk over Time")
plt.legend()
plt.grid(True)
plt.savefig("draf_Rt.png", bbox_inches="tight")
plt.close()

# 2) Domain-level risks for DC-B as example
plt.figure()
for d in domains:
    s = df[(df.DC=="DC-B") & (df.domain==d)]
    plt.plot(s.time, s.R_i, label=d)
plt.xlabel("Time")
plt.ylabel("Domain Risk R_i")
plt.title("DC-B Domain Risks over Time")
plt.legend()
plt.grid(True)
plt.savefig("dcB_domain_risks.png", bbox_inches="tight")
plt.close()

# Compute static R_T (baseline) per DC using initial params (no updates)
static_results = []
for dc in data_centers:
    R_i_vals = {}
    for d in domains:
        P0,I0,D0 = init_params[dc][d]     # initial params exactly as used by dynamic
        R_i = (P0 * I0) / D0
        R_i_vals[d] = R_i
    R_T_static = sum(weights[d]*R_i_vals[d] for d in domains)
    static_results.append({"DC": dc, "R_T_static": R_T_static,
                           "R_C": R_i_vals["C"], "R_P": R_i_vals["P"],
                           "R_E": R_i_vals["E"], "R_H": R_i_vals["H"]})

# Convert to dataframe and save
import pandas as pd
df_static = pd.DataFrame(static_results)
df_static.to_csv("draf_static_baseline.csv", index=False)
print(df_static)

# after df and df_static exist
summary_rows = []
for dc in data_centers:
    static_val = float(df_static[df_static.DC==dc]["R_T_static"])
    dyn = df[(df.DC==dc) & (df.domain=="TOTAL")]["R_i"]
    mean_dyn = dyn.mean()
    std_dyn = dyn.std()
    peak_dyn = dyn.max()
    duration_above = (dyn > 0.8).sum()  # 0.8 threshold örnek
    perc_impr = 100.0 * (static_val - mean_dyn) / static_val
    summary_rows.append({"DC":dc, "R_T_static":static_val, "mean_R_T_dynamic":mean_dyn,
                         "std_dynamic":std_dyn, "peak_dynamic":peak_dyn,
                         "duration_above_0.8":duration_above, "perc_improvement":perc_impr})
pd.DataFrame(summary_rows).to_csv("draf_comparison_summary.csv", index=False)
# ---------------------------------------------
# PLOT 1: Static vs Dynamic time-series overlay
# ---------------------------------------------
plt.figure()
for dc in data_centers:
    s_dyn = df[(df.DC==dc) & (df.domain=="TOTAL")]
    static_val = float(df_static[df_static.DC==dc]["R_T_static"])
    plt.plot(s_dyn.time, s_dyn.R_i, label=f"{dc} Dynamic", linewidth=2)
    plt.hlines(static_val, xmin=0, xmax=time_steps-1, colors='gray', linestyles='dashed', label=f"{dc} Static")

plt.xlabel("Time Step")
plt.ylabel("Aggregated Risk R_T")
plt.title("Static vs Dynamic Risk Comparison")
plt.legend()
plt.grid(True)
plt.savefig("figure_static_vs_dynamic.png", bbox_inches="tight")
plt.close()

print("✅ Saved: figure_static_vs_dynamic.png")

# ---------------------------------------------
# PLOT 2: Mean R_T static vs dynamic (bar chart)
# ---------------------------------------------
summary_df = pd.read_csv("draf_comparison_summary.csv")

x = np.arange(len(summary_df["DC"]))  # DC indices
width = 0.35

plt.figure()
plt.bar(x - width/2, summary_df["R_T_static"], width, label="Static", color="lightgray")
plt.bar(x + width/2, summary_df["mean_R_T_dynamic"], width, label="Dynamic", color="skyblue")

for i, v in enumerate(summary_df["perc_improvement"]):
    plt.text(x[i], max(summary_df["R_T_static"][i], summary_df["mean_R_T_dynamic"][i]) * 1.02,
             f"-{v:.1f}%", ha='center', fontsize=9, color="black")

plt.xticks(x, summary_df["DC"])
plt.ylabel("Mean Aggregated Risk R_T")
plt.title("Mean Static vs Dynamic Risk per Data Center")
plt.legend()
plt.grid(axis="y", linestyle=":")
plt.savefig("figure_mean_bar.png", bbox_inches="tight")
plt.close()

print("✅ Saved: figure_mean_bar.png")


print("Simulation complete. CSV and PNG files saved in current directory.")
