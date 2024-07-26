#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import re
import os

#%%
# Open and read the file
def load_tml(filename):
    residues = []
    frame = []
    time = []
    codes = []
    #id = filename.split('-')[0]
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split() 
            if len(parts) < 3:  
                continue
            residues.append(int(parts[0])) 
            frame_number = int(parts[-2])
            frame.append(frame_number)
            time.append(frame_number)
            codes.append(parts[-1])  
        df = pd.DataFrame({
        'residue': residues,
        'frame': frame, 
        'time': time,
        'code': codes
        })

        df['timestep'] = df['frame'].apply(lambda x: 2000 if x < 1457 else 10000 if x >= 1457 else 7880)
        df['time'] = df['time'] * df['timestep'] /2000000
        
    return df
#%%
path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis'
try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
#%%
#tri_tml = load_tml('clustered-aligned-5000f.tml')
A_tml = load_tml('chainA-1000f.tml')
B_tml = load_tml('chainB-1000f.tml')
C_tml = load_tml('chainC-1000f.tml')
#%%
path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/vmd_analysis'
os.chdir(path)
mon_tml = load_tml('monomer.tml')
#mon_tml = mon_tml[mon_tml['time']<= 10.01]
#%%
# A_tml = A_tml[~A_tml['residue'].between(49, 52)]
# B_tml = B_tml[~B_tml['residue'].between(49, 52)]
mon_oterminus_tml = mon_tml[~mon_tml['residue'].between(352, 362)]

#%%
# Sequential merging
# df_final = A_tml.merge(B_tml, on=['residue', 'time'], how='outer')
# df_final = df_final.merge(C_tml, on=['residue', 'time'], how='outer')
# df_final = df_final.merge(mon_tml, on=['residue', 'time'], how='outer')
#%%
df_same = df_final[
    (df_final['codechainA'] == df_final['codechainB']) & 
    (df_final['codechainB'] == df_final['codechainC']) & 
    (df_final['codechainC'] == df_final['codemon'])
]
#%%
df_mondiff = df_final[
    (df_final['codechainA'] == df_final['codechainB']) & 
    (df_final['codechainB'] == df_final['codechainC']) & 
    ~(df_final['codechainC'] == df_final['codemon'])
]
#%%
df_trimer = df_final.drop('codemon', axis = 1)
df_tri_same = df_final[
    (df_final['codechainA'] == df_final['codechainB']) & 
    (df_final['codechainB'] == df_final['codechainC'])
]
#%%
df_tri_diff = df_final[
    ~(df_final['codechainA'] == df_final['codechainB']) & 
    ~(df_final['codechainB'] == df_final['codechainC']) &
    (df_final['codechainA'] == df_final['codechainC'])
]
#%%
# Replace codes with number
#code_array = df['code'].unique()
#code_mapping = {code: i for i, code in enumerate(code_array)}
code_mapping = {'C': 0, 'E': 1, 'B': 2, 'T': 3, 'H': 4, 'G': 5, 'I': 6}
# For aggregated version:
#code_mapping = {'C': 0, 'G': 2, 'E': 1, 'B': 1, 'T': 1, 'H': 2, 'I': 2}
mon_tml['code'] = mon_tml['code'].replace(code_mapping)
# df_final['codechainA'] = df_final['codechainA'].replace(code_mapping)
# df_final['codechainB'] = df_final['codechainB'].replace(code_mapping)
# df_final['codechainC'] = df_final['codechainC'].replace(code_mapping)
reversed_code_mapping = {v: k for k, v in code_mapping.items()}
# For aggregated version
#reversed_code_mapping = {0: 'coil', 2: 'helix', 1: 'sheet'}
#%%
# diff_res = df_mondiff['residue']
# diff_time = df_mondiff['time']
#%%
# Pivot the DataFrame
pivoted_df = mon_tml.pivot(index='residue', columns='time', values='code')
# tri_pivotted = tri_tml.pivot(index='residue', columns='time', values='codechainA')
# A_pivotted = B_tml.pivot(index='residue', columns='time', values='codechainA')
# B_pivotted = B_tml.pivot(index='residue', columns='time', values='codechainA')

# %%
num_unique_codes = len(code_mapping)

# Create a color map with 'num_unique_codes' discrete colors from 'viridis'
viridis = plt.cm.get_cmap('viridis', num_unique_codes)  
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
plt.show()

#%%
# Open the file and process each line
def open_hbond_file(filename):
    data = []
    name = os.path.splitext(filename)[0].split('-')[0]
    with open(filename, 'r') as file:
        next(file)  # Skip the header line if there is one
        next(file)
        for line in file:
            parts = line.strip().split()  # Split line into parts
            if len(parts) == 3:  # Assuming each line is made up of three elements
                donor, acceptor, occupancy = parts
                #donor_res = int(re.search(r'(\d+)', donor).group(0))
                #acceptor_res = int(re.search(r'(\d+)', acceptor).group(0))
                occupancy = float(occupancy.rstrip('%'))# Remove '%' and convert to float
                data.append([donor, acceptor, occupancy]) #donor_res, acceptor_res])      
    hbond_df = pd.DataFrame(data, columns=['Donor', 'Acceptor', f'{name}'])#, 'Donor_res', 'Acceptor_res'])
    return hbond_df
#%%
os.chdir('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis')
hbond_A = open_hbond_file('chainA-details.dat')
hbond_B = open_hbond_file('chainB-details.dat')
hbond_C = open_hbond_file('chainC-details.dat')
os.chdir('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/vmd_analysis')
hbond_mon = open_hbond_file('mon-details.dat')

#%%
hbond_AB = pd.merge(hbond_A, hbond_B, on=['Donor', 'Acceptor'], how='outer')
hbond_ABC = pd.merge(hbond_AB, hbond_C, on=['Donor', 'Acceptor'], how='outer')
hbond_all = pd.merge(hbond_ABC, hbond_mon, on=['Donor', 'Acceptor'], how='outer')
hbond_all.fillna({'chainA': 0, 'chainB': 0, 'chainC': 0, 'mon':0}, inplace=True)
hbond_all['ABC_mean'] = hbond_all[['chainA', 'chainB', 'chainC']].mean(axis=1)
hbond_all['ABC_stdev'] = hbond_all[['chainA', 'chainB', 'chainC']].std(axis=1)
hbond_all['diff'] = hbond_all['mon'] - hbond_all['ABC_mean']
#hbond_all['ABC_stdev'] = hbond_all['ABC_stdev'].replace(0,1)
#%%
hbond_all['diff'] <= 5
# %%
threshold = 5 * hbond_all['ABC_stdev']
final_df = hbond_all[np.abs(hbond_all['diff']) > threshold]
filtered_df = final_df[~((final_df['ABC_stdev'] == 0) & (final_df['mon'] <= 5))]
#%%
pivot_df = filtered_df.pivot(index='Donor', columns='Acceptor', values='diff')
pivot_mon = hbond_mon.pivot(index='Donor', columns='Acceptor', values='mon')
#pivot_df.fillna(0, inplace=True)
occupancy_array = pivot_df.values
masked_array = np.ma.masked_equal(occupancy_array, 0)
#%%
# find sorting order of x-axis, and sort its labels
xlabels = pivot_df.columns
x_resids = [int(l.split("-")[0][3:]) for l in xlabels] # get list of integer numbers representing resids
x_sorting = np.argsort(np.array(x_resids)) # find sorting order
xlabels = xlabels[x_sorting] # re-sort x-axis labels

ylabels = pivot_df.index
# find sorting order of y-axis, and sort its labels (same as above)
y_resids = [int(l.split("-")[0][3:]) for l in ylabels]
y_sorting = np.argsort(np.array(y_resids)) 
ylabels = ylabels[y_sorting]
#%%
# sort the data
sorted_df = pivot_df.reindex(index=ylabels, columns=xlabels)
#%%
fig, ax = plt.subplots(figsize=(16, 14))
cmap = plt.cm.coolwarm  # Use a diverging colormap

# Set the color for NaN values
cmap.set_bad(color='white')  # This sets NaN values to appear white

# Create a masked array where NaNs are masked
masked_array = np.ma.masked_invalid(sorted_df.values)

# Create the grid for plotting
x_edges = np.arange(sorted_df.shape[1] + 1)
y_edges = np.arange(sorted_df.shape[0] + 1)

# Create the pcolormesh plot
c = ax.pcolormesh(x_edges, y_edges, masked_array, cmap=cmap, shading='flat', vmin=-70, vmax=70)

# Set the ticks at the center of the cells
ax.set_xticks(x_edges[:-1] + 0.5, minor=False)
ax.set_yticks(y_edges[:-1] + 0.5, minor=False)

# Set tick labels
ax.set_xticklabels(sorted_df.columns, rotation=90)
ax.set_yticklabels(sorted_df.index)

# Add a color bar
colorbar = plt.colorbar(c, ax=ax)
colorbar.set_label('difference of h-bond occupancy percentage between monomer and mean of three chains of trimer')

import re
res = []
for r in sorted_df.index:
    num = re.findall(r'\d+', r)
    res.extend(map(int, num))
# Cu_cite_type1 = [134, 175, 183, 188]
# Cu_cite_type1_color = 'orangered'

# Cu_cite_type2 = [139, 174, 329]
# Cu_cite_type2_color = 'magenta'

# for i, residue in enumerate(Cu_cite_type1):
#     if i == 0: 
#         plt.axhline(y=residue, color=Cu_cite_type1_color, linestyle='--', linewidth=1.7, label='Cu cite type 1')
#     else:
#         plt.axhline(y=residue, color=Cu_cite_type1_color, linestyle='--', linewidth=1.7)

# for i, residue in enumerate(Cu_cite_type2):
#     if i == 0:
#         plt.axhline(y=residue, color=Cu_cite_type2_color, linestyle='--', linewidth=1.7, label='Cu cite type 2')
#     else:
#         plt.axhline(y=residue, color=Cu_cite_type2_color, linestyle='--', linewidth=1.7)

# Add labels and title if necessary
ax.set_xlabel('Acceptor')
ax.set_ylabel('Donor')
ax.set_title('H-Bond Occupancy Heatmap')
ax.tick_params(axis='x', labelsize=10) 
ax.tick_params(axis='y', labelsize=10) 

plt.show()
#%%
column_max = filtered_df['diff'].max()
column_min = filtered_df['diff'].min()

print("Maximum value in column 'diff':", column_max)
print("Minimum value in column 'diff':", column_min)
#%%
max_index_A = filtered_df['diff'].idxmax()
min_index_A = filtered_df['diff'].idxmin()

print("Index of the maximum value in column 'A':", max_index_A)
print("Index of the maximum value in column 'A':", min_index_A)
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
#%%
def plot_hbond(df, column):
    pivot_df = df.pivot(index='Donor', columns='Acceptor', values=column)

    occupancy_array = pivot_df.values
    masked_array = np.ma.masked_equal(occupancy_array, 0)
    
    xlabels = pivot_df.columns
    x_resids = [int(l.split("-")[0][3:]) for l in xlabels] # get list of integer numbers representing resids
    x_sorting = np.argsort(np.array(x_resids)) # find sorting order
    xlabels = xlabels[x_sorting] # re-sort x-axis labels
    
    ylabels = pivot_df.index
    # find sorting order of y-axis, and sort its labels (same as above)
    y_resids = [int(l.split("-")[0][3:]) for l in ylabels]
    y_sorting = np.argsort(np.array(y_resids)) 
    ylabels = ylabels[y_sorting]
    # sort the data
    sorted_df = pivot_df.reindex(index=ylabels, columns=xlabels)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    cmap = plt.cm.coolwarm  # Use a diverging colormap
    
    # Set the color for NaN values
    cmap.set_bad(color='white')  # This sets NaN values to appear white
    
    # Create a masked array where NaNs are masked
    masked_array = np.ma.masked_invalid(sorted_df.values)
    
    # Create the grid for plotting
    x_edges = np.arange(sorted_df.shape[1] + 1)
    y_edges = np.arange(sorted_df.shape[0] + 1)
    
    # Create the pcolormesh plot
    c = ax.pcolormesh(x_edges, y_edges, masked_array, cmap=cmap, shading='flat', vmin=-70, vmax=70)
    
    # Set the ticks at the center of the cells
    ax.set_xticks(x_edges[:-1] + 0.5, minor=False)
    ax.set_yticks(y_edges[:-1] + 0.5, minor=False)
    
    # Set tick labels
    ax.set_xticklabels(sorted_df.columns, rotation=90)
    ax.set_yticklabels(sorted_df.index)
    
    # Add a color bar
    colorbar = plt.colorbar(c, ax=ax)
    colorbar.set_label('difference of h-bond occupancy percentage between monomer and mean of three chains of trimer')
    
    # Add labels and title if necessary
    ax.set_xlabel('Acceptor')
    ax.set_ylabel('Donor')
    ax.set_title('H-Bond Occupancy Heatmap')
    ax.tick_params(axis='x', labelsize=10) 
    ax.tick_params(axis='y', labelsize=10) 
    
    plt.show()
# %%
plot_hbond(hbond_mon, 'finalmon')
# %%
