import csv

# input_file = './Apr/groundTruth_scenario_5_496.csv'
# output_file = './Apr/groundTruth_converted_5_496.csv'

input_file = './Normalized_groundTruth_scenario_8_res100.csv'
output_file = './Normalized_groundTruth_converted_8_res100.csv'

# input_file = './Apr/normalized_groundTruth_scenario_7.csv'
# output_file = './Apr/groundTruth_converted_7.csv'

def is_number(s):
    
    try:
        float(s)
        return True
    except ValueError:
        return False

print(f"Converting {input_file} ...")

with open(input_file, 'r', newline='') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    
    # 1. Read and Write Header
    header = next(reader)
    # Write header exactly as is
    f_out.write(",".join(header) + "\n")
    
    # Setup writer for data lines (Auto-quote strings like paths, leave numbers alone)
    writer = csv.writer(f_out, quoting=csv.QUOTE_NONNUMERIC)
    
    for row in reader:
        if not row: continue
        
        # --- Line 1: Data (Columns 1-6) ---
        # Format: Length, Risk, Time, PathX, PathY, Fitness
        data_row = []
        for i in range(6):
            val = row[i]
            # Convert numbers to floats so they don't get quoted
            if is_number(val):
                data_row.append(float(val))
            else:
                data_row.append(val)
        writer.writerow(data_row)
        
        # --- Line 2: Weights (Column 7) ---
        # Format: "w1;w2;w3" (Quoted)
        weight_str = row[6]
        writer.writerow([weight_str])

print(f"Success! Saved to {output_file}")













































