#%%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import MDAnalysis as mda
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
