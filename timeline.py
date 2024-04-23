#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import re
import os
#%%
path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'
try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
#%%
# Initialize lists to hold your column data
residues = []
time = []
codes = []
strided_num = 50
#%%
# Open and read the file
with open('chainC-1000f.tml', 'r') as file:
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
plt.title('Timeline plot of chain C')
plt.show()

# %%
# Initialize an empty list to store the data
data = []

# Open the file and process each line
with open('hbonds-details-3000.dat', 'r') as file:
    next(file)  # Skip the header line if there is one
    next(file)
    for line in file:
        parts = line.strip().split()  # Split line into parts
        if len(parts) == 3:  # Assuming each line is made up of three elements
            donor, acceptor, occupancy = parts
            donor_res = int(re.search(r'(\d+)', donor).group(0))
            acceptor_res = int(re.search(r'(\d+)', acceptor).group(0))
            occupancy = float(occupancy.rstrip('%'))# Remove '%' and convert to float
            data.append([donor, acceptor, occupancy, donor_res, acceptor_res])
            

# Create a DataFrame
hbond_df = pd.DataFrame(data, columns=['Donor', 'Acceptor', 'Occupancy', 'Donor_res', 'Acceptor_res'])

print(hbond_df)

# %%
print(hbond_df['Occupancy'].describe())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(hbond_df['Occupancy'], bins=10, kde=True)
plt.title('Histogram of H-Bond Occupancies')
plt.xlabel('Occupancy')
plt.ylabel('Frequency')
plt.show()

# %%
from sklearn.cluster import KMeans

# Assume we want to cluster the data into low, medium, and high occupancy bonds
kmeans = KMeans(n_clusters=3, random_state=0).fit(hbond_df[['Occupancy']])
hbond_df['cluster'] = kmeans.labels_

# Plotting hbond_df
sns.scatterplot(data=hbond_df, x='Donor', y='Occupancy', hue='cluster', palette='viridis')
plt.title('Clustering of H-Bonds by Occupancy')
plt.show()

# %%
import numpy as np

# Randomly assign secondary structure elements
np.random.seed(0)
df['secondary_structure'] = np.random.choice(['α-helix', 'β-sheet', 'random coil'], size=len(df))

# Analyzing occupancy by secondary structure
print(df.groupby('secondary_structure')['Occupancy'].mean())

sns.boxplot(data=df, x='secondary_structure', y='Occupancy')
plt.title('Occupancy by Secondary Structure')
plt.show()
