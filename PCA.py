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
#%%
u_1 = mda.Universe('chainA.gro', 'chainA_cluster_aligned.xtc')
non_hydrogen_atoms = u_1.select_atoms('not type H', 'resid 53:100')
with mda.Writer('new_trajectory.xtc', non_hydrogen_atoms.n_atoms) as W:
    for ts in u_1.trajectory:
        W.write(non_hydrogen_atoms)
#specific_residues = u_1.select_atoms("resid 53:100")
#%%
u_2 = mda.Universe('chainB.gro', 'chainB_cluster_aligned.xtc', 'chainC_cluster_aligned.xtc')
with MDAnalysis.Writer("u1.pdb", multiframe=True) as W:
    for ts in u_1.trajectory:
        W.write(calphas)
# %%
# Load the univers
u = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/5UE6_newbox.gro', '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/')
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
path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/vmd_analysis'
try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
#%%
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

# %%
# Set dir path
path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'
 
try:
    os.chdir(path)
    print(f"Successfully changed the working directory to {path}")
except Exception as e:
    print(f"Error occurred while changing the working directory: {e}")
#%%
# Load CA, noh, of all four trajectories and align them to the first frame of monomer
#u_mon = mda.Universe('mon-1841f-6ms-res53to362-noh-CA.pdb')
u_A = mda.Universe('chainA-CA-res53-noh.gro', 'chainA-1002f-res53-CA-noh.dcd')
u_B = mda.Universe('chainB-CA-res53-noh.gro', 'chainB-1002f-res53-CA-noh.dcd')
u_C = mda.Universe('chainC-CA-res53-noh.gro', 'chainC-1002f-res53-CA-noh.dcd')
u_mon = mda.Universe('mon-CA-res53-noh.gro', 'mon-3017f-res53-noh-CA.dcd')
#%%
# Select all atoms except hydrogen atoms
def selection(u):
    non_hydrogen_atoms = u.select_atoms('not name H*')
# From the non-hydrogen selection, select only alpha carbon atoms
    alpha_carbons = non_hydrogen_atoms.select_atoms('resid 53:362 and name CA')
    alpha_carbon_universe = mda.Merge(alpha_carbons)
    print(f"Number of all atoms: {len(u.atoms)}")
    print(f"Number of noh: {len(non_hydrogen_atoms.atoms)}")
    print(f"Number of CA and noh: {len(alpha_carbon_universe.atoms)}")
    return(alpha_carbon_universe)
#%%
# u_A = selection(u_A)
# u_B = selection(u_B)
# u_C = selection(u_C)
#%%
#%%
# u_mon = selection(u_mon)
#%%

#%%
ref = u_mon
ref.trajectory[0]
#%%
from MDAnalysis.analysis.align import AlignTraj

# def align_to_reference(universe, ref):
#     align.AlignTraj(universe, ref, select='all', filename = f'{universe}_aligned.xtc').run()

# # Align all universes to the reference frame
# align_to_reference(u_mon, ref)
# align_to_reference(u_A, ref)
# align_to_reference(u_B, ref)
# align_to_reference(u_C, ref)

#%%
align.AlignTraj(u_A, ref, select='all', filename = 'chainA_CA_noh_res53_aligned.xtc').run()
align.AlignTraj(u_B, ref, select='all', filename = 'chainB_CA_noh_res53_aligned.xtc').run()
align.AlignTraj(u_C, ref, select='all', filename = 'chainC_CA_noh_res53_aligned.xtc').run()
align.AlignTraj(u_mon, ref, select='all', filename = 'mon_CA_noh_res53_aligned.xtc').run()

#%%

u.trajectory[-1]  # set mobile trajectory to last frame
ref.trajectory[0]  # set reference trajectory to first frame

u_ca = u.select_atoms('name CA')
ref_ca = ref.select_atoms('name CA')
unaligned_rmsd = rms.rmsd(u_ca.positions, ref_ca.positions, superposition=False)
print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
#%%
import biobox as bb
M = bb.Molecule()
M.import_pdb('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/monomer.pdb')

# %%
pc = M.pca(components = 2)
# %%
PC_values = np.arange(10) + 1
plt.plot(PC_values, (pc[1].explained_variance_ratio_)*100, 'o-', linewidth=2, color='purple')
plt.xlabel('Principal Component')
plt.ylabel('Percentage explained variance')
plt.show()

# %%
#pc = pca(n_components=2, svd_solver='arpack').fit(M)
# %%
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


projection = pc[0]
#%%
# Generate labels for each class
#labels = np.array(['monomer']*3017 + ['A']*1002 + ['B']*1002 + ['C']*1002)
#labels = np.array(['A']*1000 + ['B']*1000 + ['C']*1000)
labels = np.array(['monomer']*1841)
#labels = np.array(['A']*999)
#labels = np.array(['B']*1000 + ['C']*1000)
#labels = np.array(['trimer']*3000)


# Colors for each class
#color_map = { 'monomer': cm.RdPu, 'A': cm.Blues, 'B': cm.YlGn, 'C': cm.YlOrBr}
#color_map = {'monomer': 'purple', 'A': 'darkcyan', 'B': 'yellowgreen', 'C': 'chocolate'}
#color_map = {'A': cm.Blues, 'B': cm.YlGn, 'C': cm.YlOrBr}
color_map = {'monomer': 'purple'}
#color_map = {'A': cm.Blues, 'B': cm.YlGn, 'C': cm.Oranges}
#color_map = {'A': 'orange'}
#color_map = {'B': 'blue', 'C': 'green'}
#color_map = {'trimer':'Greys'}

# Plot each class
for label in np.unique(labels):
    # Indices for elements of the current class
    indices = np.where(labels == label)[0]
    # Scatter plot for the class
    #color_array = np.linspace(0, 1, indices.size)
    #plt.scatter(projection[indices,0], projection[indices,1], c=color_array,  cmap=color_map[label], label=label)
    #color_array = np.linspace(0, 1, indices.size)
    #plt.scatter(projection[indices,0], projection[indices,1], c=color_array,  cmap=color_map[label])
    #plt.scatter(projection[indices,0], projection[indices,1], color = 'black', alpha = 0.4)
    plt.scatter(projection[indices,0], projection[indices,1], c=color_map[label] , alpha = 0.4)

# legend_patches = [
#     mpatches.Patch(color=color_map['A'](0.6), label='Chain A'),   # Using 0.9 to get a dark shade
#     mpatches.Patch(color=color_map['B'](0.4), label='Chain B'),
#     mpatches.Patch(color=color_map['C'](0.65), label='Chain C'),
#     mpatches.Patch(color=color_map['monomer'](0.9), label='monomer')
# ]

# legend_patches = [
#     mpatches.Patch(color='darkcyan', label='Chain A'),   # Using 0.9 to get a dark shade
#     mpatches.Patch(color='yellowgreen', label='Chain B'),
#     mpatches.Patch(color='chocolate', label='Chain C'),
#     mpatches.Patch(color='purple', label='monomer')
# ]


# Add legend and labels
#plt.legend(handles=legend_patches, frameon=False)
plt.xlabel(f'PC1 ({round(pc[1].explained_variance_ratio_[0]*100,1)}%)')
plt.ylabel(f'PC2 ({round(pc[1].explained_variance_ratio_[1]*100,1)}%)')
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
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0)
kmeans.fit(projection)
# %%
plt.scatter(projection[:,0], projection[:,1], c = kmeans.labels_)

# %%
from sklearn.metrics import silhouette_samples, silhouette_score
silhouette_avg = silhouette_score(projection, kmeans.labels_)

# %%

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score


X = projection[:,0]
y = projection[:,1]
range_n_clusters = [2, 3]
 #%%
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

# %%
from sklearn.cluster import DBSCAN

results_dict = {}
cutoff = (3,4)
density = (9,10)

for e in cutoff:
    for d in density:
        clustering = DBSCAN(eps = e, min_samples = d).fit(projection)
        labels = clustering.labels_
        n_clusters = len(set(labels)) -1
        noise = list(labels).count(-1)

        if e not in results_dict:
            results_dict[e] = {}
        results_dict[e][d] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'noise': noise
        }

        print(f'eps={e}, min_samples={d}: number of clusters = {n_clusters}, detected outliers = {noise}')


# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'projection' is your dataset used for DBSCAN fitting

fig, axes = plt.subplots(len(cutoff), len(density), figsize=(10, 10))  # Adjust the figure size as needed
axes = axes.flatten()  # Flatten the axes array for easy iteration if it's 2D

# Plotting
plot_index = 0
for eps, dens_dict in results_dict.items():
    for min_samples, data in dens_dict.items():
        labels = data['labels']
        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            # Extract core and non-core samples
            xy = projection[class_member_mask & core_samples_mask]
            axes[plot_index].plot(
                xy[:, 0],
                xy[:, 1],
                'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=10
            )

            xy = projection[class_member_mask & ~core_samples_mask]
            axes[plot_index].plot(
                xy[:, 0],
                xy[:, 1],
                'o',
                markerfacecolor=tuple(col),
                markeredgecolor='k',
                markersize=6
            )

        axes[plot_index].set_title(f'eps={eps}, min_samples={min_samples}, Clusters: {data["n_clusters"]}')
        plot_index += 1

# Adjust layout
plt.tight_layout()
plt.show()

# %%
non_noise_mask = labels != -1
filtered_projection = projection[non_noise_mask]
# %%
# %%
clustering = DBSCAN(eps =2.5, min_samples = 14).fit(filtered_projection)
labels = clustering.labels_
n_clusters = len(set(labels))
noise = list(labels).count(-1)

#%%
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = filtered_projection[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

    xy = filtered_projection[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
plt.xlabel(f'PC1 ({round(pc[1].explained_variance_ratio_[0] * 100, 1)}%)')
plt.ylabel(f'PC2 ({round(pc[1].explained_variance_ratio_[1] * 100, 1)}%)')
#plt.title(f"Estimated number of clusters: {n_clusters-1}")
plt.show()

# %%
