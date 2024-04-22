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
u = mda.Universe('chainC.gro', 'chainC_cluster.xtc')
ref = mda.Universe('chainC.gro', 'chainC_cluster.xtc')

u.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

u_ca = u.select_atoms('name CA')
ref_ca = ref.select_atoms('name CA')
unaligned_rmsd = rms.rmsd(u_ca.positions, ref_ca.positions, superposition=False)
print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
# %%
# Align the trajectory and save the coordinates
aligner = align.AlignTraj(u, ref, select='protein',
                           filename='chainC_cluster_aligned.xtc').run()
#%%
# Load the aligned trajectory and calculate rmsd
u_aligned = mda.Universe('chainC.gro', 'chainC_cluster_aligned.xtc')
ref_aligned = mda.Universe('chainC.gro', 'chainC_cluster_aligned.xtc')

u_aligned.trajectory[-1]  # set mobile trajectory to last frame
ref_aligned.trajectory[0]  # set reference trajectory to first frame

u_aligned_ca = u_aligned.select_atoms('name CA')
ref_aligned_ca = ref_aligned.select_atoms('name CA')
aligned_rmsd = rms.rmsd(u_aligned_ca.positions, ref_aligned_ca.positions, superposition=False)

print(f"Aligned RMSD: {aligned_rmsd:.2f}")

# Calculate RMSD
R = MDAnalysis.analysis.rms.RMSD(u_aligned, ref_aligned, select="backbone")
R.run()
rmsd = R.rmsd.T
df = pd.DataFrame(R.rmsd, columns=['Frame', 'Time (ns)', 'Backbone'])
#%%
# RMSD plot 
ax = df.plot(x='Frame', y=['Backbone'], kind='line')
ax.set_ylabel(r'RMSD ($\AA$)')
ax.set_title("RMSD of clustered chain C")
ax.text(15000, 3.2, f"RMSD is {aligned_rmsd:.2f}", bbox = dict(facecolor = 'white', alpha = 0.3))
# %%
# Calculate RMSF
c_alphas = u_aligned.select_atoms('protein and name CA')
R = rms.RMSF(c_alphas).run()
#%%
# RMSF plot
plt.plot(c_alphas.resids, R.results.rmsf)
plt.title("RMSF of clustered chain C")
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
# %%
# Plot RMSF only for a subset of residues 
start_residue = 55  # example start residue
end_residue = 350   # example end residue

# Filter resids and RMSF values for the specified range
indices = (c_alphas.resids >= start_residue) & (c_alphas.resids <= end_residue)
selected_resids = c_alphas.resids[indices]
selected_rmsf = R.results.rmsf[indices]

# Plot the RMSF for the selected range of residues
plt.plot(selected_resids, selected_rmsf)
plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
plt.show()
# %%
# Load the strided coordinates  
u_strided = mda.Universe('data/5ue6_newbox.gro', 'final_1000/aligned_1000.xtc', stride=100)

#%%
# Create pairwise rmsd matrix
matrix = diffusionmap.DistanceMatrix(u_strided, select='backbone').run()
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
M.import_pdb('data/5ue6_newbox.pdb')
# %%
dist = M.rmsd_distance_matrix(flat=True)
# %%
M.coordinates.shape
# %%
