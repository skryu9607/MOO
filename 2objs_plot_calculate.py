import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_ground_truth(csv_file):
    """
    Loads Ground Truth solutions (u_T) from CSV.
    """
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        cols = ['Length', 'Risk', 'TravelTime']
        df_clean = df.dropna(subset=cols)
        S_gt = df_clean[cols].values.astype(float)
        print(f"Loaded Ground Truth: {S_gt.shape[0]} solutions.")
        return S_gt
    except Exception as e:
        print(f"Error loading Ground Truth CSV: {e}")
        return None

def load_algo_log(txt_file):
    """
    Loads BMRPS log file.
    """
    try:
        df = pd.read_csv(txt_file, sep=r'\s*,\s*', engine='python')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading Log TXT: {e}")
        return None

def calculate_regret_drain_bmrps_weights(S_gt, df_log):
    # 1. Extract Test Weights directly from BMRPS Log
    w_cols = [c for c in df_log.columns if c.strip() in ['w1', 'w2', 'w3']]
    if len(w_cols) < 3:
        print("Error: Weight columns (w1, w2, w3) not found in log.")
        return [], [], []
    
    # Use unique weights found in the log as the "Test Set"
    logged_weights = df_log[w_cols].values.astype(float)
    test_weights = np.unique(logged_weights, axis=0)
    print(f"Evaluated on {len(test_weights)} unique weights from BMRPS log.")

    # 2. Pre-compute Optimal Costs (Ground Truth Baseline)
    # Cost_opt(w) = min_{s in S_gt} (w . s)
    all_gt_costs = test_weights @ S_gt.T
    opt_costs = np.min(all_gt_costs, axis=1)

    regret_history = []
    rel_regret_history = []
    iterations = []
    S_alg = None
    
    unique_iters = sorted(df_log['Iteration'].unique())
    obj_cols = ['f1', 'f2', 'f3']

    for it in unique_iters:
        # Get new solutions found in this iteration
        batch_rows = df_log[df_log['Iteration'] == it]
        new_sols = batch_rows[obj_cols].values.astype(float)
        
        # Accumulate solutions
        if S_alg is None:
            S_alg = new_sols
        else:
            S_alg = np.vstack((S_alg, new_sols))
            
        # 3. Compute Current Algorithm Costs
        all_alg_costs = test_weights @ S_alg.T
        curr_costs = np.min(all_alg_costs, axis=1)
        
        # 4. Calculate Regret
        diff = curr_costs - opt_costs
        max_regret = np.max(diff)
        
        # Relative Regret
        epsilon = 1e-6
        rel_diff = diff / (opt_costs + epsilon)
        max_rel_regret = np.max(rel_diff)
        
        regret_history.append(max_regret)
        rel_regret_history.append(max_rel_regret)
        iterations.append(it)
        
    return iterations, regret_history, rel_regret_history

# --- Main Execution ---
if __name__ == "__main__":
    gt_file = './morrf_results/results_sub_231_iterations_40000.csv'
    log_file = 'RPS_log_batch_scenario_1_size_2.txt'
    
    S_gt = load_ground_truth(gt_file)
    df_log = load_algo_log(log_file)
    
    if S_gt is not None and df_log is not None:
        iters, reg, rel_reg = calculate_regret_drain_bmrps_weights(S_gt, df_log)
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(iters, reg, marker='o')
        ax1.set_title('Max Regret (Evaluated on BMRPS Weights)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Absolute Regret')
        ax1.grid(True)
        
        ax2.plot(iters, rel_reg, marker='o', color='orange')
        ax2.set_title('Max Relative Regret (Evaluated on BMRPS Weights)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Relative Regret')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
