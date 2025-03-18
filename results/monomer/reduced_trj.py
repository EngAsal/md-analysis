#%%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from MDAnalysis.coordinates.XTC import XTCWriter
from MDAnalysis.analysis import hydrogenbonds as hbonds
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align, diffusionmap, rms
# import nglview as nv

import warnings
# suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore')
#%%
u_aligned = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5UE6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/aligned_1000.xtc')

# %%
n = 20
u_reduced = u_aligned.trajectory[::n]
#%%
# Create a new XTC writer for the reduced trajectory
# Make sure to specify the correct number of atoms and, optionally, the frame rate
with XTCWriter('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/reduced_trajectory.xtc', n_atoms=u_aligned.atoms.n_atoms) as writer:
    # Iterate over the trajectory with the defined stride
    for ts in u_aligned.trajectory[::n]:
        writer.write(u_aligned.atoms)
# %%
# Load the reduced trajectory
u_reduced = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5UE6_newbox.gro', 
                         '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/reduced_trajectory.xtc')
#%%
matrix = diffusionmap.DistanceMatrix(u_reduced, select='backbone').run()
#%%
np.savetxt('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/rmsd_matrix.csv', matrix.dist_matrix, delimiter=',')
# %%
plt.imshow(matrix.dist_matrix, cmap='viridis')
plt.xlabel('Frame')
plt.ylabel('Frame')
plt.colorbar(label=r'RMSD ($\AA$)')
# %%
# Assume matrix.dist_matrix is your RMSD matrix from the MDAnalysis DistanceMatrix
# Replace 'matrix.dist_matrix' with the actual variable containing your matrix if it's named differently
rmsd_matrix = matrix.dist_matrix

# %%
mean_rmsd_per_frame = np.mean(rmsd_matrix, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(mean_rmsd_per_frame, label='Mean RMSD per Frame')
plt.xlabel('Frame')
plt.ylabel('Mean RMSD (Å)')
plt.title('Mean RMSD per Frame Over Time')
plt.legend()
plt.show()
#%%
stabilization_threshold = 0.001  # for example, 0.01 Å RMSD change
stabilized_frame = None
for i in range(1, len(mean_rmsd_per_frame)):
    if abs(mean_rmsd_per_frame[i] - mean_rmsd_per_frame[i-1]) < stabilization_threshold:
        stabilized_frame = i
        break

if stabilized_frame is not None:
    print(f"The RMSD stabilizes at frame {stabilized_frame}.")
else:
    print("The RMSD does not stabilize within the given threshold.")
# %%
# Define the starting frame (frame 238 in your case)
starting_frame = 238

# Create a new Universe with the reduced trajectory, skipping the first 238 frames
with mda.Writer('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/stablized_trajectory.xtc', u_reduced.atoms.n_atoms) as W:
    for ts in u_reduced.trajectory[starting_frame:]:
        W.write(u_reduced)
# %%
hydrogens = u_reduced.select_atoms('type H')  # Adjust the selection if your hydrogen atoms are named differently
donors = hydrogens.bonded_atoms.select_atoms('type N or type O')  # Assumes hydrogens are bonded to N or O
acceptors = u_reduced.select_atoms('type O or type N')

# %%
