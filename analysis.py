#%%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from MDAnalysis.coordinates.XTC import XTCWriter
import MDAnalysis as mda
from MDAnalysis.analysis import pca, diffusionmap, rms, align
from MDAnalysis.coordinates.PDB import PDBWriter
import MDAnalysis.analysis.rms
import biobox as bb
import inspect

#%%
mon = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/md_center_sliced_100.xtc')

mon_CA_noh_res53 = mon.select_atoms('protein and name CA and not name H and resid 53:362')
# Write the topology file
mon_CA_noh_res53.write('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon_CA_noh_res53.pdb')
# Write the trajectory file
with MDAnalysis.Writer("/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon_CA_noh_res53.xtc", mon_CA_noh_res53.n_atoms) as W:
    for ts in mon.trajectory:
        W.write(mon_CA_noh_res53)
# Load the universe for CA-noh residues 53-362
mon_CA_noh_res53 = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon_CA_noh_res53.pdb', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon_CA_noh_res53.xtc')
monomer_frames = len(mon_CA_noh_res53.trajectory)
#%%
chainA = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA.gro', '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_cluster_sliced_10.xtc')
chainA_CA_noh_res53 = chainA.select_atoms('protein and name CA and not name H and resid 53:362')
chainA_CA_noh_res53.write('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_CA_noh_res53.pdb')
with MDAnalysis.Writer("/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_CA_noh_res53.xtc", chainA_CA_noh_res53.n_atoms) as W:
    for ts in chainA.trajectory:
        W.write(chainA_CA_noh_res53)
chainA_CA_noh_res53 = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_CA_noh_res53.pdb', '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_CA_noh_res53.xtc')
chainA_frames = len(chainA_CA_noh_res53.trajectory)
#%%
# Set the reference as the monomer universe
ref = mon_CA_noh_res53
ref.trajectory[0]  
# Important: select in the align function selects atoms for alignment and not for writing the trajectory with those atoms
align.AlignTraj(chainA_CA_noh_res53, ref, select='all', filename = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_CA_noh_res53_aligned.xtc').run()
#%%
# Load multi pdb with the aligned universe for chain A (on monomer)
mon_A = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon_CA_noh_res53.pdb', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon_CA_noh_res53.xtc', 
                     '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/chainA_CA_noh_res53_aligned.xtc')
#%%
with MDAnalysis.Writer("/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/mon_A.pdb", multiframe=True) as W:
    for ts in mon_A.trajectory:
        W.write(mon_A)
#%%
pca_analysis('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/multipdb.pdb', n_components = 2, monomer_frames = 1840, chainA_frames = 5001, show_sequence = True)
#%%
all_rmsf_values = []
mean_rmsf = []
sd_rmsf = []
upper_limit =[]
lower_limit = []


for u in chainB_dict, chainC_dict:
    molecule_name , universe = next(iter(u.items()))
    backbone = universe.select_atoms('protein and backbone')
    rmsf_values = rms.RMSF(backbone).run().rmsf
    print(f"{molecule_name} RMSF values shape: {rmsf_values.shape}")
    all_rmsf_values.append(rmsf_values)
    #calculate mean RMSF of all chains for each residue
for res in range(len(rmsf_values)):
    mean_rmsf.append(np.mean([res_rmsf[res] for res_rmsf in all_rmsf_values]))
    sd_rmsf.append(np.std([res_rmsf[res] for res_rmsf in all_rmsf_values]))
    upper_limit.append(mean_rmsf[res] + sd_rmsf[res])
    lower_limit.append(mean_rmsf[res] - sd_rmsf[res])
plt.plot(backbone.resids, mean_rmsf)
plt.fill_between(backbone.resids, lower_limit, upper_limit, color='grey', alpha=0.7)
#%
monomer_frames=3000
chainA_frames=1000
chainB_frames=1000
chainC_frames=1000
#%%
# Merge DataFrames
dfs_list = list(hbond_dfs.values())
hbond_all = dfs_list[0]
for df in dfs_list[1:]:
    hbond_all = pd.merge(hbond_all, df, on=['Donor', 'Acceptor'], how='outer')
    
    # Fill hbond occupancy for residues not having hbond with 0
    for chain in hbond_dfs.keys():
        hbond_all.fillna({chain: 0}, inplace=True)

    # Calculate mean, standard deviation, and difference
    chains = list(hbond_dfs.keys())
    if 'monomer' in chains:
        chains.remove('monomer')
        hbond_all['ABC_mean'] = hbond_all[chains].mean(axis=1)
        hbond_all['ABC_stdev'] = hbond_all[chains].std(axis=1)
        hbond_all['diff'] = hbond_all['monomer'] - hbond_all['ABC_mean']
    else:
        hbond_all['ABC_mean'] = hbond_all[chains].mean(axis=1)
        hbond_all['ABC_stdev'] = hbond_all[chains].std(axis=1)
#%%
os.chdir('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer')
u = mda.Universe('data/5ue6_newbox.gro', 'final_3000/final_3000_aligned.xtc')
ref = mda.Universe('data/5ue6_newbox.gro', 'final_3000/final_3000_aligned.xtc')
#%%

# u.trajectory[-1]  # set mobile trajectory to last frame
# ref.trajectory[0]  # set reference trajectory to first frame

# u_ca = u.select_atoms('name CA')
# ref_ca = ref.select_atoms('name CA')
# unaligned_rmsd = rms.rmsd(u_ca.positions, ref_ca.positions, superposition=False)
# print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
# # Align the trajectory and save the coordinates
aligner = align.AlignTraj(u, ref, select='protein and resid 55:350',
                           filename='final_3_aligned.xtc').run()
# # Load the aligned trajectory and calculate rmsd
# u_aligned = mda.Universe('data/ABC_newbox.gro', 'cluster_aligned.xtc')
# ref_aligned = mda.Universe('data/ABC_newbox.gro', 'cluster_aligned.xtc')

# u_aligned.trajectory[-1]  # set mobile trajectory to last frame
# ref_aligned.trajectory[0]  # set reference trajectory to first frame

# u_aligned_ca = u_aligned.select_atoms('name CA')
# ref_aligned_ca = ref_aligned.select_atoms('name CA')
# aligned_rmsd = rms.rmsd(u_aligned_ca.positions, ref_aligned_ca.positions, superposition=False)

# print(f"Aligned RMSD: {aligned_rmsd:.2f}")
#%%

# # Assuming the residue numbers start from 51 to 362
# residue_numbers = np.arange(55, 350)

# Create a plot for the mean RMSF values
plt.plot(c_alphas_A.resids, R_A.results.rmsf, label = 'chain A', color = 'darkcyan')
plt.plot(c_alphas_B.resids, R_B.results.rmsf, label = 'chain B', color = 'yellowgreen')
plt.plot(c_alphas_C.resids, R_C.results.rmsf, label = 'chain C', color = 'chocolate')
plt.plot(residue_numbers, mean_rmsf_per_residue, label='Mean RMSF', color = 'black')
plt.fill_between(residue_numbers, lower_limit, upper_limit, color = 'grey', alpha = 0.7)
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
plt.legend()
#%%
# # Calculate RMSD
# R = MDAnalysis.analysis.rms.RMSD(u_aligned, ref_aligned, select="backbone")
# R.run()
# rmsd = R.rmsd.T
# df = pd.DataFrame(R.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
# df[r'Time ($\mu$s)'] = df[r'Time ($\mu$s)'] / 1000000
# #%%
# # RMSD plot 
ax = df.plot(x=r'Time ($\mu$s)', y=['Backbone'], kind='line', legend=False, color='black')
ax.set_ylabel(r'RMSD ($\AA$)')
#%%
# Plot rmsd of all chains in one plot
os.chdir('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains')
u_A = mda.Universe('chainA.gro', 'chainA_cluster_sliced_10.xtc')
ref_A = mda.Universe('chainA.gro', 'chainA_cluster_sliced_10.xtc')
# u_A.trajectory[-1]  # set mobile trajectory to last frame
ref_A.trajectory[0]  # set reference trajectory to first frame
R_A = MDAnalysis.analysis.rms.RMSD(u_A, ref_A, select="backbone")
R_A.run()
rmsd_A = R_A.rmsd.T

u_B = mda.Universe('chainB.gro', 'chainB_cluster_sliced_10.xtc')
ref_B = mda.Universe('chainB.gro', 'chainB_cluster_sliced_10.xtc')
u_B.trajectory[-1]  # set mobile trajectory to last frame
ref_B.trajectory[0]  # set reference trajectory to first frame
R_B = MDAnalysis.analysis.rms.RMSD(u_B, ref_B, select="backbone")
R_B.run()
rmsd_B = R_B.rmsd.T

u_C = mda.Universe('chainC.gro', 'chainC_cluster_sliced_10.xtc')
ref_C = mda.Universe('chainC.gro', 'chainC_cluster_sliced_10.xtc')
u_C.trajectory[-1]  # set mobile trajectory to last frame
ref_C.trajectory[0]  # set reference trajectory to first frame
R_C = MDAnalysis.analysis.rms.RMSD(u_C, ref_C, select="backbone")
R_C.run()
rmsd_C = R_C.rmsd.T
#%%
#os.chdir()
u_mon = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro',
                      '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/md_center.xtc')
ref_mon = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro',
                      '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/md_center.xtc')
# u_mon.trajectory[-1]  # set mobile trajectory to last frame
ref_mon.trajectory[0]  # set reference trajectory to first frame
R_mon = MDAnalysis.analysis.rms.RMSD(u_mon, ref_mon, select="backbone")
R_mon.run()
rmsd_mon = R_mon.rmsd.T
df_mon = pd.DataFrame(R_mon.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
plt.plot(df_mon[r'Time ($\mu$s)'], df_mon['Backbone'], color = 'purple',  alpha = 0.9, label = 'monomer')
#%%
df_A = pd.DataFrame(R_A.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_B = pd.DataFrame(R_B.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_C = pd.DataFrame(R_C.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_mon = pd.DataFrame(R_mon.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_A[r'Time ($\mu$s)'] = df_A[r'Time ($\mu$s)'] / 1000000
df_B[r'Time ($\mu$s)'] = df_B[r'Time ($\mu$s)'] / 1000000
df_C[r'Time ($\mu$s)'] = df_C[r'Time ($\mu$s)'] / 1000000
df_mon[r'Time ($\mu$s)'] = df_mon[r'Time ($\mu$s)'] / 1000000
#RMSD plot 
plt.plot(df_A[r'Time ($\mu$s)'], df_A['Backbone'], color = 'darkcyan', alpha = 0.9, label = 'chain A')
plt.plot(df_B[r'Time ($\mu$s)'], df_B['Backbone'], color = 'yellowgreen',  alpha = 0.9, label = 'chain B')
plt.plot(df_C[r'Time ($\mu$s)'], df_C['Backbone'], color = 'chocolate',  alpha = 0.9, label = 'chain C')
plt.plot(df_mon[r'Time ($\mu$s)'], df_mon['Backbone'], color = 'purple',  alpha = 0.9, label = 'monomer')
plt.ylabel(r'RMSD ($\AA$)')
plt.xlabel(r'Time ($\mu$s)')
plt.legend()

# %%
# # Calculate RMSF
# c_alphas = u_aligned.select_atoms('protein and backbone')
# R = rms.RMSF(c_alphas).run()
# #%%
# # RMSF plot
# plt.plot(c_alphas.resids, R.results.rmsf, color='purple')
# plt.xlabel('Residue number')
# plt.ylabel('RMSF ($\AA$)')
# # %%
# # Plot RMSF only for a subset of residues 
# start_residue = 55  # example start residue
# end_residue = 349   # example end residue

# # Filter resids and RMSF values for the specified range
# indices = (c_alphas.resids >= start_residue) & (c_alphas.resids <= end_residue)
# selected_resids = c_alphas.resids[indices]
# selected_rmsf = R.results.rmsf[indices]

# # Plot the RMSF for the selected range of residues
# plt.plot(selected_resids, selected_rmsf, color='purple')
# plt.xlabel('Residue number')
# plt.ylabel('RMSF ($\AA$)')
# plt.show()
# %%
# # Load the strided coordinates  
# u_strided = mda.Universe('data/5ue6_newbox.gro', 'final_3000_sliced_1000.xtc')

#%%
# Create pairwise rmsd matrix
matrix = diffusionmap.DistanceMatrix(u_strided, select='name CA').run()
#%%
# Save the matrix
#np.savetxt('/final_1000/rmsd_matrix.csv', matrix.dist_matrix, delimiter=',')
# Show the matrix
plt.imshow(matrix.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')

# %%
monomer = u_aligned.select_atoms('chain A')
# %%
import biobox as bb
# %%
M = bb.Molecule()
M.import_pdb('strided50_aligned.pdb')
# %%
dist = M.rmsd_distance_matrix(flat=True)
#%%
pairwise_rmsd = M.rmsd_distance_matrix()
#%%
plt.imshow(pairwise_rmsd, cmap = 'viridis')
plt.colorbar(label = 'RMSD ($\AA$)')
plt.title('Pairwise RMSD of monomer')
plt.xlabel('Frame')
# plt.ylabel('Frame')
# %%
# M.coordinates.shape
# %%
#Plot RMSF for chains
#Load the aligned trajectory and calculate rmsd
os.chdir('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer')
u_aligned_mon = mda.Universe('5ue6_newbox.gro', 'final_center_sliced_100.xtc')
selection_str = 'resid 53:348'
u_aligned_mon = u_aligned_mon.select_atoms(selection_str)
c_alphas_mon = u_aligned_mon.select_atoms('protein and name CA')
R_mon = rms.RMSF(c_alphas_mon).run()
R_mon_values = R_mon.rmsf
rmsf_values_mon = R_mon.rmsf

#%%
os.chdir('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC')

u_aligned_A = mda.Universe('chainA.gro', 'chainA_cluster_sliced_50.xtc')
selection_str = 'resid 53:348'
u_aligned_A = u_aligned_A.select_atoms(selection_str)


u_aligned_B = mda.Universe('chainB.gro', 'chainB_cluster_sliced_50.xtc')
u_aligned_B = u_aligned_B.select_atoms(selection_str)

u_aligned_C = mda.Universe('chainC.gro', 'chainC_cluster_sliced_50.xtc')
u_aligned_C = u_aligned_C.select_atoms(selection_str)

# Calculate RMSF
c_alphas_A = u_aligned_A.select_atoms('protein and name CA')
c_alphas_B = u_aligned_B.select_atoms('protein and name CA')
c_alphas_C = u_aligned_C.select_atoms('protein and name CA')


R_A = rms.RMSF(c_alphas_A).run()
R_B = rms.RMSF(c_alphas_B).run()
R_C = rms.RMSF(c_alphas_C).run()

R_A_values = R_A.rmsf
R_B_values = R_B.rmsf
R_C_values = R_C.rmsf

rmsf_values_a = R_A.rmsf
rmsf_values_b = R_B.rmsf
rmsf_values_c = R_C.rmsf

# Calculate the mean RMSF values per residue
mean_rmsf_per_residue = np.mean([rmsf_values_a, rmsf_values_b, rmsf_values_c], axis=0)
sd_rmsf_per_residue = np.std([rmsf_values_a, rmsf_values_b, rmsf_values_c], axis=0)
upper_limit = mean_rmsf_per_residue + sd_rmsf_per_residue
lower_limit = mean_rmsf_per_residue - sd_rmsf_per_residue

# Assuming the residue numbers start from 51 to 362
residue_numbers = np.arange(53, 349)

Cu_cite_type1 = [134, 175, 183, 188]
Cu_cite_type2 = [139, 174, 329]

Cu_cite = 'orangered'

# Create a plot for the mean RMSF values
# plt.plot(c_alphas_A.resids, R_A.results.rmsf, label = 'chain A', color = 'darkcyan')
# plt.plot(c_alphas_B.resids, R_B.results.rmsf, label = 'chain B', color = 'yellowgreen')
# plt.plot(c_alphas_C.resids, R_C.results.rmsf, label = 'chain C', color = 'chocolate')
plt.plot(c_alphas_mon.resids, R_mon.results.rmsf, label = 'monomer', color = 'purple')
plt.plot(residue_numbers, mean_rmsf_per_residue, label='trimer', color = 'black', linewidth=1)
plt.fill_between(residue_numbers, lower_limit, upper_limit, color = 'grey', alpha = 0.6)
# Plot vertical lines to highlight residues
for i, residue in enumerate(Cu_cite_type1):
    if i == 0: 
        plt.axvline(x=residue, color=Cu_cite, linewidth=1.5, label='Cu site type I')
    else:
        plt.axvline(x=residue, color=Cu_cite, linewidth=1.5)

for i, residue in enumerate(Cu_cite_type2):
    if i == 0:
        plt.axvline(x=residue, color=Cu_cite, linestyle='--', linewidth=2, label='Cu site type II')
    else:
        plt.axvline(x=residue, color=Cu_cite, linestyle='--', linewidth=2)

plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
# %%
# Create pairwise rmsd matri
matrix = diffusionmap.DistanceMatrix(u_aligned_A, select='backbone').run()
#%%
# Save the matrix
#np.savetxt('/final_1000/rmsd_matrix_chainA.csv', matrix.dist_matrix, delimiter=',')
# Show the matrix
plt.imshow(matrix.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
# %%

#%%
# u = mda.Universe('chainA.gro', 'chainA_cluster.xtc')
# subset = u.select_atoms('resid 53:362')  # Adjust the residue numbers as needed
# with mda.Writer('chainA-re53.xtc', subset.n_atoms) as W:
#     for ts in u.trajectory:
#         W.write(subset)

#%%
# Load the two structures
u1 = mda.Universe('chainA-re53.gro', 'chainA-re53.xtc')  # The one with more atoms
u2 = mda.Universe('chainB.gro', 'chainB.xtc')  # The one with fewer atoms
#%%
# Find all hydrogen atoms in each structure
h_u1 = u1.select_atoms('type H')
h_u2 = u2.select_atoms('type H')

# Compare the number of hydrogen atoms
print(f"Number of hydrogen atoms in structure 1: {len(h_u1)}")
print(f"Number of hydrogen atoms in structure 2: {len(h_u2)}")

# Identify missing hydrogens if any
if len(h_u1) != len(h_u2):
    print("The structures have a different number of hydrogen atoms.")
    # Further analysis can be done here to identify which hydrogens are missing
else:
    print("The structures have the same number of hydrogen atoms.")

#%%
u = mda.Universe('data/ABC.pdb')
# %%
path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer'
os.chdir(path)
# %%
u_aligned = mda.Universe('5ue6_newbox.gro', 'final_3000_aligned.xtc')
# %%
matrix = diffusionmap.DistanceMatrix(u_aligned, select='name CA').run()
#%%
start_frame = 0
step = 100

# Using the context manager to handle trajectory slicing

# Now compute the distance matrix only for the sliced trajectory
for ts in u_aligned.trajectory[::step]:
    matrix = diffusionmap.DistanceMatrix(u_aligned, select='name CA').run()

    # Visualize the results
    plt.imshow(matrix.dist_matrix, cmap='viridis')
    plt.xlabel('Frame')
    plt.ylabel('Frame')
    plt.colorbar(label=r'RMSD ($\AA$)')
    plt.show()

# %%
u_sliced = mda.Universe('5ue6_newbox.gro', 'final_3000_100sliced.xtc')
# %%
matrix = diffusionmap.DistanceMatrix(u_sliced, select='name CA').run()
# %%
plt.imshow(matrix.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
# %%
time_step = 20 # Time step in picoseconds per frame
start_frame = 0  # Starting frame
step = 100  # Step for slicing
num_frames = len(u_aligned.trajectory)  # Total number of frames in the trajectory

 
total_time = num_frames * step * 20 / 100000000
plt.imshow(matrix.dist_matrix, cmap='viridis', extent = [0,total_time, 0, total_time])

# Assuming 'matrix.dist_matrix' is your distance matrix calculated as previously described
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'Time ($\mu$s)')
plt.colorbar(label=r'RMSD ($\AA$)')
plt.show()
# %%
path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer'
os.chdir(path)
import pandas as pd
df = pd.read_csv('density.xvg', sep='\s+', header=None, names=['step','density'])
plt.plot(df['step'], df['density'])  
plt.xlabel('Simulation step')
plt.ylabel('density (kg/m^3)')
plt.show()
# %%
stride_step = 100
monomer_res = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/updated_final_center_aligned_res53_349.xtc')
with XTCWriter('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/aligned_res53_349_sliced100.xtc', n_atoms=monomer_res.atoms.n_atoms) as writer:
    for ts in monomer_res.trajectory[::stride_step]:
        writer.write(monomer_res.atoms)
# %%
monomer_res_100 = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/aligned_res53_349_sliced100.xtc')
# %%
universe = monomer_res_100
reference = universe
universe.trajectory[-1]
reference.trajectory[0]
R = mda.analysis.rms.RMSD(universe, reference, select='backbone').run()
df = pd.DataFrame(R.rmsd, columns=['Frame', r'Time (ps)', 'Backbone'])
df[r'Time ($\mu$s)'] = df[r'Time (ps)'] / 1000000

plt.plot(df[r'Time ($\mu$s)'], df['Backbone'], color='purple', alpha=0.9)
# %%
matrix = diffusionmap.DistanceMatrix(monomer_res_100, select='name CA').run()
# Show the matrix
plt.imshow(matrix.dist_matrix, cmap='viridis', origin='lower', vmin=0, vmax=7)
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
# %%
# Calculate RMSF
c_alphas = monomer_res_100.select_atoms('protein and backbone')
R = rms.RMSF(c_alphas).run()
# Plot RMSF only for a subset of residues 
start_residue = 49  # example start residue
end_residue = 362   # example end residue

# Filter resids and RMSF values for the specified range
indices = (c_alphas.resids >= start_residue) & (c_alphas.resids <= end_residue)
selected_resids = c_alphas.resids[indices]
selected_rmsf = R.results.rmsf[indices]

# Plot the RMSF for the selected range of residues
plt.plot(selected_resids, selected_rmsf, color='purple')
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
plt.show()
# %%
os.chdir('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer')
mobile = mda.Universe('5ue6_newbox.gro', 'updated_final_center.xtc')
ref = mda.Universe('5ue6_newbox.gro', 'updated_final_center.xtc')

mobile.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

mobile_ca = mobile.select_atoms('name CA')
ref_ca = ref.select_atoms('name CA')
unaligned_rmsd = rms.rmsd(mobile_ca.positions, ref_ca.positions, superposition=False)
print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
#%%
aligner = align.AlignTraj(mobile, ref, select='protein and resid 53:349',
                          filename='aligned_to_first_frame_res.xtc').run()
mobile = mda.Universe('5ue6_newbox.gro', 'aligned_to_first_frame_res.xtc')

#%%
mobile.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

mobile_ca = mobile.select_atoms('name CA')
ref_ca = ref.select_atoms('name CA')
aligned_rmsd = rms.rmsd(mobile_ca.positions, ref_ca.positions, superposition=False)

print(f"Aligned RMSD: {aligned_rmsd:.2f}")
#%%
