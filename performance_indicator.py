import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from pymoo.indicators.hv import HV

def is_pareto_front(points):
    """
    Correctly identify non-dominated solutions
    """
    points = np.array(points)
    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            # Check if j dominates i
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_pareto[i] = False
                break
    
    return is_pareto

def calculate_metrics(ground_truth_file, rps_log_file):
    print(f"Starting Evaluation for {rps_log_file}...\n")
 
    gt_df = pd.read_csv(ground_truth_file, skipinitialspace=True)
    
    for col in ['Length', 'Risk', 'TravelTime']:
        gt_df[col] = pd.to_numeric(gt_df[col], errors='coerce')
        
    gt_df = gt_df.dropna(subset=['Length', 'Risk', 'TravelTime'])
    
    P_star_norm = gt_df[['Length', 'Risk', 'TravelTime']].values.astype(float)
    P_star = P_star_norm[is_pareto_front(P_star_norm)]
    
    valid_solutions = []
    with open(rps_log_file, 'r') as f:
        lines = f.readlines()
        
        header = [col.strip() for col in lines[0].split(',')]
        idx_f1, idx_f2, idx_f3 = header.index('Cost_Distance'), header.index('Cost_Risk'), header.index('Cost_Time')
        idx_dup = header.index('is_duplicate') if 'is_duplicate' in header else -1
        
        for line in lines[1:]:
            if not line.strip(): continue 
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) > max(idx_f1, idx_f2, idx_f3):

                f1 = float(parts[idx_f1])
                f2 = float(parts[idx_f2])
                f3 = float(parts[idx_f3])
                
                is_dup = 0
                if idx_dup != -1 and len(parts) > idx_dup:
                    is_dup = int(float(parts[idx_dup]))
                
                if is_dup == 0:
                    valid_solutions.append([f1, f2, f3])

    RPS_non_pareto = np.array(valid_solutions, dtype=float)
    RPS_pareto = RPS_non_pareto[is_pareto_front(RPS_non_pareto)]
    
    print(f"Data Loaded: {len(P_star)} Ground Truth points, {len(RPS_pareto)} RDG solutions.")
    
    # IGD 
    min_val = np.min(P_star, axis=0)
    max_val = np.max(P_star, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0 

    P_star_norm = (P_star - min_val) / range_val
    RPS_norm = (RPS_pareto - min_val) / range_val
    RPS_norm = np.clip(RPS_norm, 0.0, 1.0)

    distances = cdist(P_star_norm, RPS_norm, metric='euclidean')
    igd_value = np.mean(np.min(distances, axis=1))
    print(f"IGD: {igd_value:.6f}")

    # Hypervolume Ratio (HVR)
    ref_point = np.array([1.1, 1.1, 1.1])
    hv_calc = HV(ref_point=ref_point)
    hv_true = hv_calc(P_star_norm)
    hv_approx = hv_calc(RPS_norm)
    hvr_value = hv_approx / hv_true if hv_true > 0 else 0
    print(f"HVR: {hvr_value:.6f}")
    
    # Regret 
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(3), size=10000)
    regrets = np.zeros(10000)
    
    for i, w in enumerate(weights):
        c_star = np.min(np.dot(P_star, w))
        c_hat = np.min(np.dot(RPS_pareto, w))
        regrets[i] = c_hat - c_star
        
    expected_regret = np.mean(regrets)
    max_regret = np.max(regrets)
    print(f"Expected Regret: {expected_regret:.6f}")
    print(f"Exact Max Regret: {max_regret:.6f}")
    
    # Spread
    Corner_Cases = RPS_pareto[:3]
    RPS_pareto_inner = RPS_pareto[3:] 
    dist_matrix = squareform(pdist(RPS_pareto,metric='euclidean'))
    N = np.size(RPS_pareto, 0)
    np.fill_diagonal(dist_matrix, np.inf)
    
    d_i = np.min(dist_matrix, axis=1)
    d_bar = np.mean(d_i)
    sum_deviation = np.sum(np.abs(d_i - d_bar))
    
    d_e = np.min(cdist(RPS_pareto_inner, Corner_Cases), axis=0)
    sum_d_e = np.sum(d_e)
    
    ExclusiveSpread = (sum_d_e + sum_deviation) / (sum_d_e + N * d_bar)
    
    sum_d_e_inclusive = 0.0
    InclusiveSpread = (sum_d_e_inclusive + sum_deviation) / (sum_d_e_inclusive + N * d_bar)
    
    print(f"Number of Pareto Points: {N}")
    print(f"Exclusive Spread: {ExclusiveSpread:.6f}")
    print(f"Inclusive Spread: {InclusiveSpread:.6f}")
    
    # Pareto Efficiency 
    ParetoRatio = np.sum(is_pareto_front(RPS_non_pareto)) / len(RPS_non_pareto) if len(RPS_non_pareto) > 0 else 0
    print(f"Pareto Ratio: {ParetoRatio:.6f}\n")  
    
    # Return everything as a dictionary
    return {
        'IGD': igd_value, 
        'HVR': hvr_value, 
        'ExpectedRegret': expected_regret, 
        'MaxRegret': max_regret, 
        'ParetoRatio': ParetoRatio,
        'ExclusiveSpread': ExclusiveSpread,
        'InclusiveSpread': InclusiveSpread,
        'ParetoPointsCount': N
    }

if __name__ == "__main__":
    gt_file = "groundTruth_converted_2.csv"
    #rps_file = "RPS_log_scenario_1_database.txt"
    
    rps_file = 'RPS_log_batch_scenario_2_size_4_database.txt'
    # Now you can capture the returned dictionary and use it!
    results = calculate_metrics(gt_file, rps_file)
    print("Final Dictionary Output:", results)
