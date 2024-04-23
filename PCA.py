#%%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
#%%
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
u = mda.Universe('chainA-re53-noh.gro', 'ABC-3000f-noh-aligned.dcd')
ref = mda.Universe('chainA-re53-noh.gro', 'ABC-3000f-noh-aligned.dcd')

u.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

u_ca = u.select_atoms('name CA')
ref_ca = ref.select_atoms('name CA')
unaligned_rmsd = rms.rmsd(u_ca.positions, ref_ca.positions, superposition=False)
print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
# %%
# Align the trajectory and save the coordinates
aligner = align.AlignTraj(u, ref, select='protein',
                           filename='ABC-aligned.xtc').run()
#%%
import biobox as bb
# %%
# Set dir path
path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'

try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
#%%
M = bb.Molecule()
M.import_pdb('ABC-CA-aligned.pdb')

# %%
pc = M.pca(components = 2)
# %%
PC_values = np.arange(2) + 1
plt.plot(PC_values, pc[1].explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# %%
#pc = pca(n_components=2, svd_solver='arpack').fit(M)
# %%
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


projection = pc[0]

# Generate labels for each class
#labels = np.array(['monomer']*2918 + ['A']*1000 + ['B']*1000 + ['C']*1000)
labels = np.array(['A']*1000 + ['B']*1000 + ['C']*1000)
#labels = np.array(['monomer']*2918)
#labels = np.array(['A']*999)
#labels = np.array(['B']*1000 + ['C']*1000)

# Colors for each class
#color_map = {'A': 'orange', 'B': 'blue', 'C': 'green', 'monomer':'purple'}
color_map = {'A': cm.Reds_r, 'B': cm.Blues_r, 'C': cm.Greens_r}
#color_map = {'monomer':'purple'}
#color_map = {'A': 'orange'}
#color_map = {'B': 'blue', 'C': 'green'}

# Plot each class
for label in np.unique(labels):
    # Indices for elements of the current class
    indices = np.where(labels == label)[0]
    # Scatter plot for the class
    color_array = color_array = np.linspace(0, 1, indices.size)
    plt.scatter(projection[indices,0], projection[indices,1], c=color_array,  cmap=color_map[label], label=label)
legend_patches = [
    mpatches.Patch(color=color_map['A'](0.1), label='Chain A'),   # Using 0.9 to get a dark shade
    mpatches.Patch(color=color_map['B'](0.1), label='Chain B'),
    mpatches.Patch(color=color_map['C'](0.1), label='Chain C')
]

# Add legend and labels
plt.legend(handles=legend_patches, title="Class", frameon=False)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot of Chain A, B and C')
plt.show()
# %%
# Load each PDB file
structure1_atoms = load_structure('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/mon-2918f-noh.pdb')
structure2_atoms = load_structure('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/multipdb.pdb')

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
