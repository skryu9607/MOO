import csv
import pandas as pd
import numpy as np

filename = "results.csv"

rows = []
header = None

with open(filename, "r") as f:
    reader = csv.reader(f)

    lines = list(reader)

i = 0
while i < len(lines):

    row = lines[i]

    # header 처리
    if row[0] == "Length":
        header = row + ["Weights"]   # add Weights column
        i += 1
        continue

    # main data row
    main_row = row  # already parsed by csv.reader → correct column count

    # next line is weights
    weight_row = lines[i + 1]
    weight = weight_row[0].replace('"', '')  # "0;1;0" → 0;1;0

    # append weight
    main_row = main_row + [weight]

    rows.append(main_row)

    i += 2


# build dataframe
df = pd.DataFrame(rows, columns=header)

# convert numerics
numeric_cols = ["Length", "Risk", "TravelTime", "Fitness"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# convert weights
df["W"] = df["Weights"].apply(lambda w: np.array([float(x) for x in w.split(";")]))

# cost vectors
cost_vectors = df[["Length", "Risk", "TravelTime"]].values

# target weights
w1  = np.array([1.0, 0.0, 0.0])
w90 = np.array([0.9, 0.1, 0.0])
w80 = np.array([0.8, 0.2, 0.0])

def find_idx(w):
    idxs = df.index[df["W"].apply(lambda x: np.allclose(x, w))]
    return idxs[0]

i1  = find_idx(w1)
i90 = find_idx(w90)
i80 = find_idx(w80)

c1  = cost_vectors[i1]
c90 = cost_vectors[i90]
c80 = cost_vectors[i80]

d10_90 = c90 - c1
d10_80 = c80 - c1

print("===== COST VECTORS =====")
print("C(1,0,0):     ", c1)
print("C(0.9,0.1,0): ", c90)
print("C(0.8,0.2,0): ", c80)

print("\n===== DIFF =====")
print("Δ(1→0.9): ", d10_90)
print("Δ(1→0.8): ", d10_80)
