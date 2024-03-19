#%%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import MDAnalysis as mda
from MDAnalysis.coordinates.chain import ChainReader
from MDAnalysis.core.universe import Merge
from MDAnalysis.analysis import pca, align, diffusionmap, rms
# import nglview as nv

import warnings
# suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore')
# %%
# Load the univers
u = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5UE6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/reduced_trajectory.xtc')
# Define the starting frame (frame 238 in your case)
starting_frame = 238

# Create a new Universe with the reduced trajectory, skipping the first 238 frames
with mda.Writer('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/stablized_trajectory.xtc', u.atoms.n_atoms) as W:
    for ts in u.trajectory[starting_frame:]:
        W.write(u)
u_stabalized = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5UE6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/stablized_trajectory.xtc')
# %%
pc = pca.PCA(u_stabalized, select='name CA',
             align=True, 
             mean=None,
             n_components=None).run()
#%%
backbone = u_stabalized.select_atoms('name CA')
n_bb = len(backbone)
print('There are {} alpha carbon atoms in the analysis'.format(n_bb))
print(pc.p_components.shape)
print(f"PC1: {pc.variance[0]:.5f}") 
for i in range(3):
    print(f"Cumulated variance: {pc.cumulated_variance[i]:.3f}")
# %%
plt.plot(pc.cumulated_variance[:10])
plt.xlabel('Principal component')
plt.ylabel('Cumulative variance')
plt.show()
#%%
# Reduced dimensional space
transformed = pc.transform(backbone, n_components=3)
#%%
df = pd.DataFrame(transformed,
                  columns=['PC{}'.format(i+1) for i in range(3)])
df['Time (ps)'] = df.index * u_aligned.trajectory.dt
df.head()

#%%
# Save the principal components (eigenvectors)
np.savetxt('pca_components.dat', pc.p_components)

# Save the eigenvalues
np.savetxt('pca_eigenvalues.dat', pc.eigenvalues)

# Save the mean structure (average structure over the trajectory)
np.savetxt('pca_mean_structure.dat', pc.mean)



# Assuming `projection` is your data projected onto the principal components
# and the first two columns correspond to PC1 and PC2, respectively

# Plot PC1 vs PC2
plt.figure(figsize=(10, 7))  # Set the figure size as needed
plt.scatter(projection[:, 0], projection[:, 1], alpha=0.5)  # Adjust alpha for point transparency

plt.title('PCA: Projection onto PC1 and PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)  # Optional: adds a grid for easier visualization

# %%
# Set dir path
import os
path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'
try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
#Load universe for chains A, B and C
chainA = mda.Universe('chainA.gro', 'chainA_cluster.xtc')
chainB = mda.Universe('chainB.gro', 'chainB_cluster.xtc')
chainC = mda.Universe('chainC.gro', 'chainC_cluster.xtc')
# %%
import MDAnalysis as mda
from MDAnalysis.coordinates.chain import ChainReader

# Create a Universe using the first structure/topology file
u = mda.Universe('chainC.gro', ChainReader(['chainC_cluster.xtc', 'chainB_cluster.xtc']))

# %%
u1 = mda.Universe('data/chainB.pdb', 'chainB_cluster_aligned.xtc')
u2 = mda.Universe('data/chainC.pdb', 'chainC_cluster_aligned.xtc')
#%%

# Create a new trajectory with all frames from u1 followed by all frames from u2
combined_trajectory = ChainReader([u1.trajectory, u2.trajectory])

# Merge the universes
# Note: This assumes the topology is identical for both universes.
# You would typically use the same topology file for both universes here.
merged_universe = mda.Merge(u1.atoms, u2.atoms)
merged_universe.trajectory = combined_trajectory
# %%
from Bio.PDB import PDBParser
from collections import Counter

def load_structure(file_name):
    parser = PDBParser()
    structure = parser.get_structure(file_name, file_name)
    atoms = [atom.get_name() for atom in structure.get_atoms()]
    return Counter(atoms)

# Load each PDB file
structure1_atoms = load_structure('data/chainA.pdb')
structure2_atoms = load_structure('data/chainC.pdb')

# Compare the two sets of atoms
if structure1_atoms == structure2_atoms:
    print("The PDB files have the same atoms.")
else:
    print("The PDB files do NOT have the same atoms.")
    missing_in_1 = structure2_atoms - structure1_atoms
    missing_in_2 = structure1_atoms - structure2_atoms
    if missing_in_1:
        print("Atoms missing in file1:", missing_in_1)
    if missing_in_2:
        print("Atoms missing in file2:", missing_in_2)

# %%
