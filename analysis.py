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
import MDAnalysis.analysis.rms
import biobox as bb
import inspect
#%%
# Load the aligned universe and stride it if needed
def load_universe(molecule_name, top_file, trj_file, stride_step=None):
    if molecule_name == 'monomer':
        base_path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer'

    elif molecule_name == 'trimer' or 'chainA' or 'chainB' or 'chainC':
        base_path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains'

    else:
        print(f"Unsupported molecule: {molecule_name}")
        return
    
    top_file_path = os.path.join(base_path, top_file)
    trj_file_path = os.path.join(base_path, trj_file)

    aligned_trj_file = f"{os.path.splitext(trj_file_path)[0]}_aligned.xtc"
    if os.path.exists(aligned_trj_file):
        print('Aligned trajectory found. Loading the aligned trajectory file.')
        traj_to_load = aligned_trj_file
    else:
        print('Aligned trajectory file does not exist. Proceeding with alignment.')
        universe = mda.Universe(top_file_path, trj_file_path)
        reference = mda.Universe(top_file_path, trj_file_path)
        print('Aligning the trajectory. This may take a while.')
        aligner = align.AlignTraj(universe, reference, select='protein', filename=aligned_trj_file).run()
        print('Trajectory aligned and saved.')
        traj_to_load = aligned_trj_file

    if stride_step is not None:
        strided_trj_file = f"{os.path.splitext(trj_file_path)[0]}_sliced_{stride_step}.xtc"
        if os.path.exists(strided_trj_file):
            print(f'Strided trajectory with {stride_step} steps found. Loading the strided trajectory file.')
            traj_to_load = strided_trj_file
        else:
            print('Strided trajectory file does not exist. Proceeding with striding.')
            universe = mda.Universe(top_file_path, aligned_trj_file)
            output_filename = strided_trj_file
            with XTCWriter(output_filename, n_atoms=universe.atoms.n_atoms) as writer:
                for ts in universe.trajectory[::stride_step]:
                    writer.write(universe.atoms)
            traj_to_load = output_filename

    universe = mda.Universe(top_file_path, traj_to_load)
    print("Aligned universe loaded.")
    return {molecule_name: universe, 'stride_step': stride_step}

def get_variable_names(args):
    frame = inspect.currentframe().f_back.f_back
    names = {id(v): k for k, v in frame.f_locals.items()}
    return [names.get(id(arg), None) for arg in args]

# Plot RMSD for one or more universes
def rmsd_plot(*universes, colors):
    labels = get_variable_names(universes)

    if len(universes) != len(colors):
        raise ValueError('Error: Number of colors does not match the number of universes.')
    
    for universe, color, label in zip(universes, colors, labels):
        reference = universe
        universe.trajectory[-1]
        reference.trajectory[0]
        R = mda.analysis.rms.RMSD(universe, reference, select='backbone').run()
        df = pd.DataFrame(R.rmsd, columns=['Frame', r'Time ($\mu$s)', 'Backbone'])
        df[r'Time ($\mu$s)'] = df[r'Time ($\mu$s)'] / 1000000

        plt.plot(df[r'Time ($\mu$s)'], df['Backbone'], color=color, label=label, alpha=0.9)

    if len(universes) > 1:
        plt.legend(frameon=False)

    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'RMSD ($\AA$)')
    plt.show()
#%%

def rmsf_plot(*universes, molecule_type, colors, start_res=None, end_res=None, plot_chains=True, plot_mean=False):
    
    labels = get_variable_names(universes)

    if len(universes) != len(colors):
        raise ValueError('Error: Number of colors does not match the number of universes.')
    
    if len(universes) != len(molecule_type):
        raise ValueError('Error: Number of molecule types should match the number of universes.')
    universe_dict = {universes[i]: molecule_type[i] for i in range(len(universes))}

    if len(universes) == 1 and plot_mean:
        raise ValueError('Error: Cannot plot mean RMSF with only one universe.')
    
    all_rmsf_values = []
    chains_rmsf_values = []
    if universe_dict. == 'monomer':


    for u, color, label in zip(universes, colors, labels):
        backbone = u.select_atoms('protein and backbone')
        if start_res is not None and end_res is not None:
            indices = (backbone.resids >= start_res) & (backbone.resids <= end_res)
            backbone = backbone[indices]

        rmsf_values = rms.RMSF(backbone).run().rmsf
        print(f"{label} RMSF values shape: {rmsf_values.shape}")
                        
        if plot_chains and not plot_mean:
            plt.plot(backbone.resids, rmsf_values, color=color, label=label)

        if plot_mean:
            all_rmsf_values = np.array(all_rmsf_values)
            print(f"All RMSF values shape: {all_rmsf_values.shape}")
            mean_rmsf = np.mean(all_rmsf_values, axis=0)
            sd_rmsf = np.std(all_rmsf_values, axis=0)
            upper_limit = mean_rmsf + sd_rmsf
            lower_limit = mean_rmsf - sd_rmsf
            plt.plot(residue_numbers, mean_rmsf)
            plt.fill_between(residue_numbers, lower_limit, upper_limit, color='grey', alpha=0.7)
            
        if len(universes) > 1:
            plt.legend(frameon=False)
        
    plt.xlabel('Residue number')
    plt.ylabel('RMSF ($\AA$)')
    plt.show()


def pairwise_rmsd(universe_dict, timestep):
    # timestep in fs is the time between two frames. It was used to be 20 fs in the simulation.
    if universe_dict['stride_step'] is None:
        print('Error: No striding was applied when loading the universe')
        return
    else: 
        universe = universe_dict['universe']
        matrix = diffusionmap.DistanceMatrix(universe, select='name CA').run()
        stride_step = universe_dict['stride_step']
        total_time = len(universe.trajectory) * stride_step * timestep / 1000000

        plt.imshow(matrix.dist_matrix, cmap='viridis', origin='lower', extent = [0, total_time, 0, total_time], vmin=0, vmax=2)
        plt.ylabel(r'Time ($\mu$s)')
        plt.xlabel(r'Time ($\mu$s)')
        plt.colorbar(label=r'RMSD ($\AA$)')
        plt.show() 
#%%
def tml_plot(filename, molecule_type, timestep, aggregated = False):
    # timestep in fs is the time between two frames. It was used to be 20 fs in the simulation.
    residues = []
    time = []
    codes = []

    if molecule_type == 'monomer':
       base_path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/vmd_analysis'

    elif molecule_type == 'trimer':
        base_path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis'

    else:
        print(f"Unsupported molecule: {molecule_type}")
        return
    
    file_path = os.path.join(base_path, filename)

    with open(file_path, "r") as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split() 
            if len(parts) < 3:  
                continue
            residues.append(int(parts[0]))  
            time.append(timestep*0.0001*int(parts[-2]))  
            codes.append(parts[-1])  
        df = pd.DataFrame({
        'residue': residues,
        'time': time,
        'code': codes
    })
        
    if aggregated:
        code_mapping = {'C': 0, 'G': 2, 'E': 1, 'B': 1, 'T': 1, 'H': 2, 'I': 2}
        reversed_code_mapping = {0: 'coil', 2: 'helix', 1: 'sheet'}
    else:
        code_mapping = {'C': 0, 'E': 1, 'B': 2, 'T': 3, 'H': 4, 'G': 5, 'I': 6}
        reversed_code_mapping = {v: k for k, v in code_mapping.items()}

    df['code'] = df['code'].replace(code_mapping)
    pivoted_df = df.pivot(index='residue', columns='time', values='code')

    num_unique_codes = len(code_mapping)
    viridis = plt.cm.get_cmap('viridis', num_unique_codes)  
    # Sample these colors evenly across the colormap
    colors = viridis(np.linspace(0, 1, num_unique_codes))  
    cmap = ListedColormap(colors)

    # Determine the boundaries for the discrete color bar
    codes_min, codes_max = pivoted_df.min().min(), pivoted_df.max().max()
    boundaries = np.arange(codes_min-0.5, codes_max+1.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    plt.figure(figsize=(12, 6))
    c = plt.pcolormesh(pivoted_df.columns, pivoted_df.index, pivoted_df, cmap=cmap, norm=norm, shading='auto')

    Cu_cite_type1 = [134, 175, 183, 188]
    for residue in Cu_cite_type1:
        plt.axhline(y=residue, color = 'red', linestyle='--', linewidth=2)

    cb = plt.colorbar(c, ticks=np.arange(codes_min, codes_max+1))
    cb.ax.set_yticklabels([reversed_code_mapping[i] for i in range(int(codes_min), int(codes_max)+1)])

    plt.xlabel('Time (ns)')
    plt.ylabel('Residue')
    plt.show()
#%%
def hbond_matrix(filenames, all_hbonds=True):
    hbond_dfs = []
    for f in filenames:
        data = []
        chain = os.path.splitext(f)[0].split('-')[-1]
        with open(f, 'r') as file:
            next(file)  # Skip the header line if there is one
            next(file)
            for line in file:
                parts = line.strip().split()  
                if len(parts) == 3:  
                    donor, acceptor, occupancy = parts
                    occupancy = float(occupancy.rstrip('%'))
                    data.append([donor, acceptor, occupancy])  
        hbond_df = pd.DataFrame(data, columns=['Donor', 'Acceptor', chain])
        hbond_dfs.append(hbond_df)

#%%
def pca_prep(*universe_discts):
    frames = []
    CA_noh_res53 = {}
    aligned_CA_noh_res53 = {}
    
    for u_dict in universe_discts: 
        molecule_name, universe = next(iter(u_dict.items()))

        u_CA_noh_res53 = universe.select_atoms('protein and name CA and not name H and resid 53:362')
        # Write the topology file
        u_CA_noh_res53.write(f'{molecule_name}_CA_noh_res53.pdb')
        # Write the trajectory file
        with MDAnalysis.Writer(f'{molecule_name}_CA_noh_res53.xtc', u_CA_noh_res53.n_atoms) as W:
            for ts in universe.trajectory:
                W.write(u_CA_noh_res53)
        # Load the universe for CA-noh residues 53-362  
        u_CA_noh_res53 = mda.Universe(f'{molecule_name}_CA_noh_res53.pdb', f'{molecule_name}_CA_noh_res53.xtc')
        CA_noh_res53[molecule_name] = u_CA_noh_res53
       
    ref = list(CA_noh_res53.values())[0]
    ref.trajectory[0]

    first_molecule_name = next(iter(CA_noh_res53))

    # Align all universes to the reference universe
    for molecule_name, universe in CA_noh_res53.items():
        align.AlignTraj(universe, ref, select='all', filename = f'{molecule_name}_CA_noh_res53_aligned.xtc').run()
        aligned_universe = mda.Universe(f'{molecule_name}_CA_noh_res53.pdb', f'{molecule_name}_CA_noh_res53_aligned.xtc')
        aligned_CA_noh_res53[molecule_name] = aligned_universe
        #print(f'Each {CA_noh_res53.keys()} has {len(list(CA_noh_res53.values())[0].trajectory)} frames.')
        frames.append(len(universe.trajectory))

    # Combine trajectories into multi_pdb
    traj_files = [f'{molecule_name}_CA_noh_res53_aligned.xtc' for molecule_name in CA_noh_res53.keys()]
    multi_pdb = mda.Universe(f'{first_molecule_name}_CA_noh_res53.pdb', *traj_files)
    
    with MDAnalysis.Writer("/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains/testmultipdb.pdb", multiframe=True) as W:
        for ts in multi_pdb.trajectory:
            W.write(multi_pdb)
    
    return multi_pdb, frames, CA_noh_res53

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
def pca_analysis(multi_pdb, n_components, monomer_frames=None, chainA_frames=None,
                 chainB_frames=None, chainC_frames=None, elbow_plot=False, show_sequence=False):

    M = bb.Molecule()
    M.import_pdb(multi_pdb)
    pc = M.pca(components = n_components)

    if elbow_plot:
        PC_values = np.arange(n_components) + 1
        plt.plot(PC_values, (pc[1].explained_variance_ratio_)*100, 'o-', linewidth=2, color='purple')
        plt.xlabel('Principal Component')
        plt.ylabel('Percentage explained variance')
        plt.show() 
        return 
    
    projection = pc[0]    
    
    indices = {}
    start = 0
    
    if monomer_frames:
        indices['monomer'] = np.arange(start, start + monomer_frames)
        start += monomer_frames
        
    if chainA_frames:
        indices['chainA'] = np.arange(start, start + chainA_frames)
        start += chainA_frames
        
    if chainB_frames:
        indices['chainB'] = np.arange(start, start + chainB_frames)
        start += chainB_frames
        
    if chainC_frames:
        indices['chainC'] = np.arange(start, start + chainC_frames)
        start += chainC_frames

    # Create labels and colors
    # labels = np.concatenate([np.full(len(indices[key]), key) for key in indices.keys()])
    color_map = {
        'monomer': 'purple' if not show_sequence else cm.RdPu,
        'chainA': 'darkcyan' if not show_sequence else cm.Blues,
        'chainB': 'yellowgreen' if not show_sequence else cm.YlGn,
        'chainC': 'chocolate' if not show_sequence else cm.Oranges
    }

    for label, idx_array in indices.items():
        if show_sequence:
            color = np.linspace(0, 1, idx_array.size)
            plt.scatter(projection[idx_array, 0], projection[idx_array, 1], c=color, cmap=color_map[label], label=label)    
        else:
            color = color_map[label]
            plt.scatter(projection[idx_array, 0], projection[idx_array, 1], c=color, label=label)

    # # Generate legend patches
    # legend_patches = [mpatches.Patch(color=color_map[label] if not show_sequence else color_map[label](0.6), label=label) for label in indices.keys()]
    # plt.legend(handles=legend_patches, frameon=False)
    
    # Add axis labels with explained variance
    plt.xlabel(f'PC1 ({round(pc[1].explained_variance_ratio_[0] * 100, 1)}%)')
    plt.ylabel(f'PC2 ({round(pc[1].explained_variance_ratio_[1] * 100, 1)}%)')
    plt.show()
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
    
    # # Fill hbond occupancy for residues not having hbond with 0
    # for chain in hbond_dfs.keys():
    #     hbond_all.fillna({chain: 0}, inplace=True)

    # # Calculate mean, standard deviation, and difference
    # chains = list(hbond_dfs.keys())
    # if 'monomer' in chains:
    #     chains.remove('monomer')
    #     hbond_all['ABC_mean'] = hbond_all[chains].mean(axis=1)
    #     hbond_all['ABC_stdev'] = hbond_all[chains].std(axis=1)
    #     hbond_all['diff'] = hbond_all['monomer'] - hbond_all['ABC_mean']
    # else:
    #     hbond_all['ABC_mean'] = hbond_all[chains].mean(axis=1)
    #     hbond_all['ABC_stdev'] = hbond_all[chains].std(axis=1)

#%%
['darkcyan', 'yellowgreen', 'chocolate', 'purple']
#%%
monomer_dict = load_universe('monomer', '5ue6_newbox.gro', 'md_center.xtc', stride_step = 100)
monomer = monomer_dict.get('monomer')
#%%
if chainC is None:
    print("Failed to load monomer universe.")
else:
    print('Proceed with analysis for monomer')
#%%
trimer_dict = load_universe('trimer', 'ABC_newbox.gro', 'cluster_aligned.xtc', stride_step = 10)
trimer = trimer_dict.get('universe')
#%%
chainA_dict = load_universe('chainA', 'chainA.gro', 'chainA_cluster.xtc', stride_step = 10)
chainA = chainA_dict.get('chainA')
#%%
chainB_dict = load_universe('chainB', 'chainB.gro', 'chainB_cluster.xtc', stride_step = 10)
chainB = chainB_dict.get('universe')
#%%
chainC_dict = load_universe('chainC', 'chainC.gro', 'chainC_cluster.xtc', stride_step = 10)
chainC = chainC_dict.get('univrse')
#%%
rmsd_plot(monomer, colors='purple')
#%%
rmsd_plot(trimer, monomer, colors=['purple', 'black'])
#%%
rmsd_plot(trimer, colors = 'black')
#%%

chainA_dic = load_universe('trimer','chainA.gro', 'chainA_cluster.xtc', stride_step= 50)
chainB_dic = load_universe('trimer','chainB.gro', 'chainB_cluster.xtc', stride_step= 50)
chainC_dic = load_universe('trimer','chainC.gro', 'chainC_cluster.xtc', stride_step= 50)
#%%
rmsf = rmsf_plot(monomer, chainA, start_res=53, end_res=348)
#%%
#%%
#universes = [chainA, chainB, chainC]
rmsf_plot(chainA, chainB, start_res=49, end_res=350, plot_chains=False, plot_mean=True)
#%%

#%%
pairwise_rmsd(chainA_dic)
#%%
pairwise_rmsd(chainB_dic)
#%%
pairwise_rmsd(chainC_dic)
#%%
pairwise_rmsd(trimer_dict)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/vmd_analysis/monomer.tml', aggregated=True)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis/trimer.tml', aggregated=True)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis/chainA.tml', aggregated=True)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis/chainB.tml', aggregated=True)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis/chainC.tml', aggregated=True)
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
#plt.plot(c_alphas_A.resids, R_A.results.rmsf, label = 'chain A', color = 'darkcyan')
#plt.plot(c_alphas_B.resids, R_B.results.rmsf, label = 'chain B', color = 'yellowgreen')
#plt.plot(c_alphas_C.resids, R_C.results.rmsf, label = 'chain C', color = 'chocolate')
# plt.plot(residue_numbers, mean_rmsf_per_residue, label='Mean RMSF', color = 'black')
# plt.fill_between(residue_numbers, lower_limit, upper_limit, color = 'grey', alpha = 0.7)
# plt.xlabel('Residue number')
# plt.ylabel('RMSF ($\AA$)')
#plt.legend()
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
u_A.trajectory[-1]  # set mobile trajectory to last frame
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

#os.chdir()
u_mon = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro',
                      '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/md_center.xtc')
ref_mon = mda.Universe('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/5ue6_newbox.gro',
                      '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/md_center.xtc')
u_mon.trajectory[-1]  # set mobile trajectory to last frame
ref_mon.trajectory[0]  # set reference trajectory to first frame
R_mon = MDAnalysis.analysis.rms.RMSD(u_mon, ref_mon, select="backbone")
R_mon.run()
rmsd_mon = R_mon.rmsd.T
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
# # Create pairwise rmsd matrix
# matrix = diffusionmap.DistanceMatrix(u_strided, select='name CA').run()
# #%%
# # Save the matrix
# #np.savetxt('/final_1000/rmsd_matrix.csv', matrix.dist_matrix, delimiter=',')
# # Show the matrix
# plt.imshow(matrix.dist_matrix, cmap='viridis')
# plt.xlabel('Frame')
# plt.ylabel('Frame')
# plt.colorbar(label=r'RMSD ($\AA$)')

# # %%
# monomer = u_aligned.select_atoms('chain A')
# # %%
# import biobox as bb
# # %%
# M = bb.Molecule()
# M.import_pdb('strided50_aligned.pdb')
# # %%
# dist = M.rmsd_distance_matrix(flat=True)
# #%%
# pairwise_rmsd = M.rmsd_distance_matrix()
# #%%
# plt.imshow(pairwise_rmsd, cmap = 'viridis')
# plt.colorbar(label = 'RMSD ($\AA$)')
# plt.title('Pairwise RMSD of monomer')
# plt.xlabel('Frame')
# plt.ylabel('Frame')
# %%
# M.coordinates.shape
# %%
#Plot RMSF for chains
#Load the aligned trajectory and calculate rmsd
os.chdir('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer')
u_aligned_mon = mda.Universe('5ue6_newbox.gro', 'md_center_sliced_10.xtc')
selection_str = 'resid 55:349'
u_aligned_mon = u_aligned_mon.select_atoms(selection_str)
c_alphas_mon = u_aligned_mon.select_atoms('protein and name CA')
R_mon = rms.RMSF(c_alphas_mon).run()
R_mon_values = R_mon.rmsf
rmsf_values_mon = R_mon.rmsf

#%%
os.chdir('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chains')

u_aligned_A = mda.Universe('chainA.gro', 'chainA_cluster_sliced_10.xtc')
selection_str = 'resid 55:349'
u_aligned_A = u_aligned_A.select_atoms(selection_str)


u_aligned_B = mda.Universe('chainB.gro', 'chainB_cluster_sliced_10.xtc')
u_aligned_B = u_aligned_B.select_atoms(selection_str)

u_aligned_C = mda.Universe('chainC.gro', 'chainC_cluster_sliced_10.xtc')
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
mean_rmsf_per_residue = np.mean([rmsf_values_b, rmsf_values_c], axis=0)
sd_rmsf_per_residue = np.std([rmsf_values_b, rmsf_values_c], axis=0)
upper_limit = mean_rmsf_per_residue + sd_rmsf_per_residue
lower_limit = mean_rmsf_per_residue - sd_rmsf_per_residue

# Assuming the residue numbers start from 51 to 362
residue_numbers = np.arange(55, 350)

Cu_cite_type1 = [134, 175, 183, 188]
Cu_cite_type1_color = 'red'

Cu_cite_type2 = [139, 174, 329]
Cu_cite_type2_color = 'green'

# Create a plot for the mean RMSF values
# plt.plot(c_alphas_A.resids, R_A.results.rmsf, label = 'chain A', color = 'darkcyan')
# plt.plot(c_alphas_B.resids, R_B.results.rmsf, label = 'chain B', color = 'yellowgreen')
# plt.plot(c_alphas_C.resids, R_C.results.rmsf, label = 'chain C', color = 'chocolate')
plt.plot(c_alphas_mon.resids, R_mon.results.rmsf, label = 'monomer', color = 'purple')
plt.plot(residue_numbers, mean_rmsf_per_residue, label='Mean RMSF', color = 'black')
plt.fill_between(residue_numbers, lower_limit, upper_limit, color = 'grey', alpha = 0.7)
# Plot vertical lines to highlight residues
for residue in Cu_cite_type1:
    plt.axvline(x=residue, color=Cu_cite_type1_color, linestyle='--', linewidth=2)

for residue in Cu_cite_type2:
    plt.axvline(x=residue, color=Cu_cite_type2_color, linestyle='--', linewidth=2)
    
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=Cu_cite_type1_color, linestyle='--', linewidth=2, label='type 1'),
    Line2D([0], [0], color=Cu_cite_type2_color, linestyle='--', linewidth=2, label='type 2')
]

plt.xlabel('Residue number')
plt.ylabel('RMSF ($\AA$)')
plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_elements, 
           labels=plt.gca().get_legend_handles_labels()[1], frameon=False)
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
