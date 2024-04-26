#%%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.coordinates.XTC import XTCWriter
import MDAnalysis as mda
from MDAnalysis.analysis import pca, diffusionmap, rms, align
import MDAnalysis.analysis.rms
import warnings
# Suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore')
 #%%
# Set dir path
path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'
try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
# %%
### Load the universe and calculate rmsd
u = mda.Universe('data/ABC_5ue6_protein.pdb', 'cluster_aligned.xtc')
ref = mda.Universe('data/ABC_5ue6_protein.pdb', 'cluster_aligned.xtc')

u.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

u_ca = u.select_atoms('name CA')
ref_ca = ref.select_atoms('name CA')
unaligned_rmsd = rms.rmsd(u_ca.positions, ref_ca.positions, superposition=False)
print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
# %%
# Align the trajectory and save the coordinates
aligner = align.AlignTraj(u, ref, select='protein',
                           filename='final_3_aligned.xtc').run()
#%%
# Load the aligned trajectory and calculate rmsd
u_aligned = mda.Universe('data/ABC_newbox.gro', 'cluster_aligned.xtc')
ref_aligned = mda.Universe('data/ABC_newbox.gro', 'cluster_aligned.xtc')

u_aligned.trajectory[-1]  # set mobile trajectory to last frame
ref_aligned.trajectory[0]  # set reference trajectory to first frame

u_aligned_ca = u_aligned.select_atoms('name CA')
ref_aligned_ca = ref_aligned.select_atoms('name CA')
aligned_rmsd = rms.rmsd(u_aligned_ca.positions, ref_aligned_ca.positions, superposition=False)

print(f"Aligned RMSD: {aligned_rmsd:.2f}")

#%%
# Calculate RMSD
R = MDAnalysis.analysis.rms.RMSD(u_aligned, ref_aligned, select="backbone")
R.run()
rmsd = R.rmsd.T
df = pd.DataFrame(R.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df[r'Time ($\mu$s)'] = df[r'Time ($\mu$s)'] / 1000000
#%%
# RMSD plot 
ax = df.plot(x=r'Time ($\mu$s)', y=['Backbone'], kind='line', legend=False, color='black')
ax.set_ylabel(r'RMSD ($\AA$)')
#%%
# Plot rmsd of all chains in one plot
u_A = mda.Universe('chainA.gro', 'chainA_cluster_aligned.xtc')
ref_A = mda.Universe('chainA.gro', 'chainA_cluster_aligned.xtc')
u_A.trajectory[-1]  # set mobile trajectory to last frame
ref_A.trajectory[0]  # set reference trajectory to first frame
R_A = MDAnalysis.analysis.rms.RMSD(u_A, ref_A, select="backbone")
R_A.run()
rmsd_A = R_A.rmsd.T

u_B = mda.Universe('chainB.gro', 'chainB_cluster_aligned.xtc')
ref_B  = mda.Universe('chainB.gro', 'chainB_cluster_aligned.xtc')
u_B.trajectory[-1]  # set mobile trajectory to last frame
ref_B.trajectory[0]  # set reference trajectory to first frame
R_B = MDAnalysis.analysis.rms.RMSD(u_B, ref_B, select="backbone")
R_B.run()
rmsd_B = R_B.rmsd.T

u_C = mda.Universe('chainC.gro', 'chainC_cluster_aligned.xtc')
ref_C = mda.Universe('chainC.gro', 'chainC_cluster_aligned.xtc')
u_C.trajectory[-1]  # set mobile trajectory to last frame
ref_C.trajectory[0]  # set reference trajectory to first frame
R_C = MDAnalysis.analysis.rms.RMSD(u_C, ref_C, select="backbone")
R_C.run()
rmsd_C = R_C.rmsd.T

#%%
df_A = pd.DataFrame(R_A.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_B = pd.DataFrame(R_B.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_C = pd.DataFrame(R_C.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
df_A[r'Time ($\mu$s)'] = df_A[r'Time ($\mu$s)'] / 1000000
df_B[r'Time ($\mu$s)'] = df_B[r'Time ($\mu$s)'] / 1000000
df_C[r'Time ($\mu$s)'] = df_C[r'Time ($\mu$s)'] / 1000000
# RMSD plot 
plt.plot(df_A[r'Time ($\mu$s)'], df_A['Backbone'], color = 'darkcyan', alpha = 0.9, label = 'chain A')
plt.plot(df_B[r'Time ($\mu$s)'], df_B['Backbone'],  color = 'yellowgreen',  alpha = 0.9, label = 'chain B')
plt.plot(df_C[r'Time ($\mu$s)'], df_C['Backbone'],  color = 'chocolate',  alpha = 0.9, label = 'chain C')
plt.ylabel(r'RMSD ($\AA$)')
plt.legend()

# %%
# Calculate RMSF
c_alphas = u_aligned.select_atoms('protein and backbone')
R = rms.RMSF(c_alphas).run()
#%%
# RMSF plot
plt.plot(c_alphas.resids, R.results.rmsf, color='purple')
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
# %%
# Plot RMSF only for a subset of residues 
start_residue = 55  # example start residue
end_residue = 349   # example end residue

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
# Load the strided coordinates  
u_strided = mda.Universe('data/5ue6_newbox.gro', 'final_3_aligned.xtc', stride=100)

#%%
# Create pairwise rmsd matrix
matrix = diffusionmap.DistanceMatrix(u_strided, select='name CA').run()
#%%
# Save the matrix
np.savetxt('/final_1000/rmsd_matrix.csv', matrix.dist_matrix, delimiter=',')
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
plt.ylabel('Frame')
# %%
M.coordinates.shape
# %%
#Plot RMSF for chains
# Load the aligned trajectory and calculate rmsd
u_aligned_A = mda.Universe('chainA.gro', 'chainA_cluster_aligned.xtc')
selection_str = 'resid 53:351'
u_aligned_A = u_aligned_A.select_atoms(selection_str)


u_aligned_B = mda.Universe('chainB.gro', 'chainB_cluster_aligned.xtc')
u_aligned_B = u_aligned_B.select_atoms(selection_str)

u_aligned_C = mda.Universe('chainC.gro', 'chainC_cluster_aligned.xtc')
u_aligned_C = u_aligned_C.select_atoms(selection_str)
# %%
# Calculate RMSF
c_alphas_A = u_aligned_A.select_atoms('protein and name CA')
c_alphas_B = u_aligned_B.select_atoms('protein and name CA')
c_alphas_C = u_aligned_C.select_atoms('protein and name CA')

R_A = rms.RMSF(c_alphas_A).run()
R_B = rms.RMSF(c_alphas_B).run()
R_C = rms.RMSF(c_alphas_C).run()
#%%
R_A_values = R_A.rmsf
R_B_values = R_B.rmsf
R_C_values = R_C.rmsf

#%%
rmsf_values_a = R_A.rmsf
rmsf_values_b = R_B.rmsf
rmsf_values_c = R_C.rmsf

# Calculate the mean RMSF values per residue
mean_rmsf_per_residue = np.mean([rmsf_values_b, rmsf_values_c], axis=0)
sd_rmsf_per_residue = np.std([rmsf_values_b, rmsf_values_c], axis=0)
upper_limit = mean_rmsf_per_residue + sd_rmsf_per_residue
lower_limit = mean_rmsf_per_residue - sd_rmsf_per_residue

# Assuming the residue numbers start from 51 to 362
residue_numbers = np.arange(53, 352)

# Create a plot for the mean RMSF values
#plt.plot(c_alphas_A.resids, R_A.results.rmsf, label = 'chain A')
#plt.plot(c_alphas_B.resids, R_B.results.rmsf, label = 'chain B')
#plt.plot(c_alphas_C.resids, R_C.results.rmsf, label = 'chain C')
plt.plot(residue_numbers, mean_rmsf_per_residue, label='Mean RMSF', color = 'royalblue')
plt.fill_between(residue_numbers, lower_limit, upper_limit, color = 'royalblue', alpha = 0.3)
plt.title("Mean and standartd deviation of RMSF for three chains")
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
#plt.legend()
# %%
# Create pairwise rmsd matri
u = mda.Universe()
matrix = diffusionmap.DistanceMatrix(u_aligned_A, select='backbone').run()
#%%
# Save the matrix
np.savetxt('/final_1000/rmsd_matrix_chainA.csv', matrix.dist_matrix, delimiter=',')
# Show the matrix
plt.imshow(matrix.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
# %%

#%%
u = mda.Universe('chainA.gro', 'chainA_cluster.xtc')
subset = u.select_atoms('resid 53:362')  # Adjust the residue numbers as needed
with mda.Writer('chainA-re53.xtc', subset.n_atoms) as W:
    for ts in u.trajectory:
        W.write(subset)

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
def write_sliced_trajectory(universe, slice_step, output_filename):
    # Initialize the trajectory writer with the output filename and the number of atoms
    with XTCWriter(output_filename, n_atoms=universe.atoms.n_atoms) as writer:
        # Iterate over the sliced trajectory
        for ts in universe.trajectory[::slice_step]:
            writer.write(universe.atoms)  # Write the current frame to the file

# %%
write_sliced_trajectory(u_aligned, 100, 'final_3000_100sliced.xtc')
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
