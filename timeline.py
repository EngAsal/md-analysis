#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

#%%
# Initialize lists to hold your column data
residues = []
time = []
codes = []

#%%
# Open and read the file
with open('mon-allframe.tml', 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue
        parts = line.split()  # Split the line into parts
        if len(parts) < 3:  # Check if the line has at least 3 parts to avoid errors
            continue
        # Append the extracted data to the lists
        residues.append(int(parts[0]))  # First number to residue
        time.append(0.02*int(parts[-2]))  # Number after 'evalempty' (assuming 'evalempty' is always the second word)
        codes.append(parts[-1])  # Last letter to code
# Create a DataFrame from the lists
df = pd.DataFrame({
    'residue': residues,
    'time': time,
    'code': codes
})

# Display the DataFrame
print(df)
#%%
# Replace codes with number
code_array = df['code'].unique()
code_mapping = {code: i for i, code in enumerate(code_array)}
df['code'] = df['code'].replace(code_mapping)
reversed_code_mapping = {v: k for k, v in code_mapping.items()}
#%%
# Pivot the DataFrame
pivoted_df = df.pivot(index='residue', columns='time', values='code')

# %%
# Number of unique codes to set the number of discrete colors needed
num_unique_codes = len(np.unique(pivoted_df.to_numpy().flatten()))

# Create a color map with 'num_unique_codes' discrete colors from 'viridis'
viridis = plt.cm.get_cmap('viridis', num_unique_codes)  # Get 'num_unique_codes' colors from viridis
colors = viridis(np.linspace(0, 1, num_unique_codes))  # Sample these colors evenly across the colormap
cmap = ListedColormap(colors)

# Determine the boundaries for the discrete color bar
codes_min, codes_max = pivoted_df.min().min(), pivoted_df.max().max()
boundaries = np.arange(codes_min-0.5, codes_max+1.5, 1)
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

# Create the mesh plot with the 'viridis'-based discrete color bar
plt.figure(figsize=(12, 6))
c = plt.pcolormesh(pivoted_df.columns, pivoted_df.index, pivoted_df, cmap=cmap, norm=norm, shading='auto')

cb = plt.colorbar(c, ticks=np.arange(codes_min, codes_max+1))
cb.ax.set_yticklabels([reversed_code_mapping[i] for i in range(int(codes_min), int(codes_max)+1)])

plt.xlabel('Time (ns)')
plt.ylabel('Residue')
plt.title('Timeline plot')
plt.show()
