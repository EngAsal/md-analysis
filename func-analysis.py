#%%
# Import libraries
import os
import logging
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
# Load the aligned universe and stride it if needed
def load_universe(molecule_name, top_file, trj_file, alignment=None, alig_res_start=None, alig_res_end=None, stride_step=None):
    if molecule_name in ['monomer']: 
        base_path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer'

    elif molecule_name in ['trimer', 'chainA', 'chainB', 'chainC']:
        base_path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'

    else:
        logging.error(f'Unsupported molecule: {molecule_name}')
        return
    
    top_file_path = os.path.join(base_path, top_file)
    trj_file_path = os.path.join(base_path, trj_file)
    traj_to_load = trj_file_path

    universe = mda.Universe(top_file_path, trj_file_path)
    universe.trajectory[0]  
    ref = universe
    ref.trajectory[-1]  
    mobile_ca = universe.select_atoms('name CA')
    ref_ca = ref.select_atoms('name CA')
    Unaligned_rmsd = rms.rmsd(mobile_ca.positions, ref_ca.positions, superposition=False)
    print(f"Unligned RMSD: {Unaligned_rmsd:.2f}")
    
    if alignment == 'all':
        alig_res_end = 362
        if molecule_name == 'monomer' and 'chainA':
            alig_res_start = 49
        elif molecule_name == 'trimer' or 'chainB' or 'chainC':
            alig_res_start = 53
        
    elif alignment == 'domain':
        alig_res_start = 53
        alig_res_end = 349
    else:
        alig_res_start = alig_res_start
        alig_res_end = alig_res_end

    # Aligning the protein to a specific residue
    aligned_res_file = f"{os.path.splitext(trj_file_path)[0]}_aligned_res{alig_res_start}_{alig_res_end}.xtc"
    if alig_res_start and alig_res_end is not None:
        if os.path.exists(aligned_res_file):
            logging.info(f'Aligned trajectory to {alig_res_start}_{alig_res_end} found. Loading the aligned trajectory file.')
            traj_to_load = aligned_res_file
        else:
            logging.info(f'Aligned trajectory to residue {alig_res_start}_{alig_res_end} does not exist. Proceeding with alignment.')
            try:
                universe = mda.Universe(top_file_path,  trj_file_path)
                align.AlignTraj(universe, universe, select=f"name CA and resid {alig_res_start}:{alig_res_end}", filename=aligned_res_file).run()
                print(f'Trajectory aligned to residue {alig_res_start}_{alig_res_end} and saved.')
                traj_to_load = aligned_res_file
            except Exception as e:
                logging.error(f'Error occurred while aligning the trajectory to residue {alig_res_start}_{alig_res_end}: {e}')

    # Striding the trajectory
    if stride_step is not None:
        # strided_trj_file = f"{os.path.splitext(trj_file_path)[0]}_sliced_{stride_step}.xtc"
        strided_trj_file = f"{os.path.splitext(aligned_res_file)[0]}_sliced_{stride_step}.xtc"
        if os.path.exists(strided_trj_file):
            logging.info('Strided trajectory found. Loading the strided trajectory file.')
            traj_to_load = strided_trj_file
        else:
            logging.info('Strided trajectory file does not exist. Proceeding with striding.')
            try:
                universe = mda.Universe(top_file_path, traj_to_load)
                with XTCWriter(strided_trj_file, n_atoms=universe.atoms.n_atoms) as writer:
                    for ts in universe.trajectory[::stride_step]:
                        writer.write(universe.atoms)
                traj_to_load = strided_trj_file
            except Exception as e:
                logging.error(f'Error occurred while striding the trajectory: {e}')

    universe = mda.Universe(top_file_path, traj_to_load)
    
    print("Aligned universe loaded.")
    return {molecule_name: universe, 'stride_step': stride_step, 'alignment': alignment, 'aligned_res_start': alig_res_start, 'aligned_res_end': alig_res_end}

# Plot RMSD for one or more universes
def rmsd_plot(*universe_dicts, colors=None):
    
    for u in universe_dicts:
        molecule_name, universe = next(iter(u.items()))
        
        start_res = u.get('aligned_res_start')
        end_res = u.get('aligned_res_end')
        alignment = u.get('alignment')
    
        if molecule_name == 'monomer':
            if alignment == 'all':
                color = 'purple'
                label = f'{molecule_name} ({alignment})'
            elif alignment == 'domain':
                color = 'plum'
                label = f'{molecule_name} (no C-ter)'

        elif molecule_name == 'trimer':
            if alignment == 'all':
                color = 'black' 
                label = f'{molecule_name} ({alignment})'
            elif alignment == 'domain':
                color = 'grey'
                label = f'{molecule_name} (no C-ter)'

        elif molecule_name == 'chainA':
            if alignment == 'all':
                color = 'darkcyan' 
                label = f'{molecule_name} ({alignment})'
            elif alignment == 'domain':
                color = 'darkcyan'
                label = f'{molecule_name} (no C-ter)'

        elif molecule_name == 'chainB':
            if alignment == 'all':
                color = 'yellowgreen' 
                label = f'{molecule_name} ({alignment})'
            elif alignment == 'domain':
                color = 'yellowgreen'
                label = f'{molecule_name} (no C-ter)'

        elif molecule_name == 'chainC':
            if alignment == 'all':
                color = 'chocolate' 
                label = f'{molecule_name} ({alignment})'
            elif alignment == 'domain':
                color = 'chocolate'
                label = f'{molecule_name} (no C-ter)'
            
        else:
            color = colors
            label = molecule_name

        # for universe, color, label, start, end in zip(u, color, molecule_name, start_res, end_res):
        reference = universe
        universe.trajectory[-1]
        reference.trajectory[0]
        R = mda.analysis.rms.RMSD(universe, reference, select=f'backbone and resid {start_res}:{end_res}').run()
        df = pd.DataFrame(R.rmsd, columns=['Frame', r'Time (ps)', 'Backbone'])
        df[r'Time ($\mu$s)'] = df[r'Time (ps)'] / 1000000

        plt.plot(df[r'Time ($\mu$s)'], df['Backbone'], color=color, label=label, alpha=0.9)

    if len(universe_dicts) > 1:
        plt.legend(frameon=False)

    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'RMSD ($\AA$)')
    plt.show()
#%%
def rmsf_plot(*universe_dicts, start_res=None, end_res=None, plot_mean=False):


    if len(universe_dicts) == 1 and plot_mean:
        raise ValueError('Error: Cannot plot mean RMSF with only one universe.')
    
    all_rmsf_values = []
    mean_rmsf = []
    sd_rmsf = []
    upper_limit = []
    lower_limit = []    
    
    for u in universe_dicts:
        # for trimer chains : ['darkcyan', 'yellowgreen', 'chocolate', 'purple']
        molecule_name , universe = next(iter(u.items()))
        alignment = u.get('alignment')
        if molecule_name == 'monomer':
            color = 'purple'
        elif molecule_name == 'trimer':
            color = 'black' 
        elif molecule_name == 'chainA':
            color = 'darkcyan'
        elif molecule_name == 'chainB':
            color = 'yellowgreen'
        elif molecule_name == 'chainC':
            color = 'chocolate'

        backbone = universe.select_atoms('protein and backbone')
        if start_res is not None and end_res is not None:
            indices = (backbone.resids >= start_res) & (backbone.resids <= end_res)
            backbone = backbone[indices]

        rmsf_values = rms.RMSF(backbone).run().rmsf         

        if molecule_name == 'monomer':
            plt.plot(backbone.resids, rmsf_values, color=color, label=molecule_name)

        if not plot_mean:               
            plt.plot(backbone.resids, rmsf_values, color=color, label=molecule_name)

        if plot_mean and molecule_name != 'monomer':
            
            all_rmsf_values.append(rmsf_values)
            

    for res in range(len(rmsf_values)):
        mean_rmsf.append(np.mean([res_rmsf[res] for res_rmsf in all_rmsf_values]))
        sd_rmsf.append(np.std([res_rmsf[res] for res_rmsf in all_rmsf_values]))        
        upper_limit.append(mean_rmsf[res] + sd_rmsf[res])
        lower_limit.append(mean_rmsf[res] - sd_rmsf[res])
    plt.plot(backbone.resids, mean_rmsf, linestyle=':', color='black', label='trimer')
    plt.fill_between(backbone.resids, lower_limit, upper_limit, color='grey', alpha=0.5)


    Cu_cite_type1 = [134, 175, 183, 188]
    Cu_cite_type2 = [139, 174, 329]
    Cu_cite_color = 'orangered'

    for i, residue in enumerate(Cu_cite_type1):
        if i == 0: 
            plt.axvline(x=residue, color=Cu_cite_color, linewidth=1.5, label='Cu site type I')
        else:
            plt.axvline(x=residue, color=Cu_cite_color, linewidth=1.5)

    for i, residue in enumerate(Cu_cite_type2):
        if i == 0:
            plt.axvline(x=residue, color=Cu_cite_color, linestyle='--', linewidth=2, label='Cu site type II')
        else:
            plt.axvline(x=residue, color=Cu_cite_color, linestyle='--', linewidth=2)
                
        if len(universe_dicts) > 1:
            plt.legend(frameon=False, fontsize='8')
        
    plt.xlabel('Residue number')
    plt.ylabel('RMSF ($\AA$)')
    plt.show()

#%%
def pairwise_rmsd(universe_dict):
    # timestep in ps is the time between two frames. It used to be 0.002 x 10000 (nstxout) = 20 ps in the simulation.
    if universe_dict['stride_step'] is None:
        print('Error: No striding was applied when loading the universe')
        return
    else: 
        molecule, universe = next(iter(universe_dict.items()))
        timestep= list(universe_dict.values())[1]

        matrix = diffusionmap.DistanceMatrix(universe, select='name CA').run()

        n_frames = matrix.dist_matrix.shape[0]
        frame_indices = np.arange(n_frames)
        time = frame_indices * timestep / 10000  

        # Visualize the matrix with time
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix.dist_matrix, cmap='viridis', origin='lower', extent=[time[0], time[-1], time[0], time[-1]], vmin=0, vmax=7)
        plt.ylabel(r'Time ($\mu$s)')
        plt.xlabel(r'Time ($\mu$s)')
        plt.colorbar(label=r'RMSD ($\AA$)')
        plt.show() 
#%%
def tml_plot(file_path, universe_dict, aggregated = False):

    timestep = universe_dict['stride_step']

    residues = []
    frame = []
    codes = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.split() 
            if len(parts) < 3:  
                continue
            residues.append(int(parts[0])) 
            frame_number = int(parts[-2])
            frame.append(frame_number)
            codes.append(parts[-1])  

        df = pd.DataFrame({
        'residue': residues,
        'frame': frame,         
        'code': codes
        })

    # Map time values back to the DataFrame
    df['time'] = df['frame'] * timestep / 10000

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

    plt.figure(figsize=(12, 10))
    c = plt.pcolormesh(pivoted_df.columns, pivoted_df.index, pivoted_df, cmap=cmap, norm=norm, shading='auto')

    Cu_cite_type1 = [134, 175, 183, 188]
    Cu_cite_type2 = [139, 174, 329]
    Cu_cite_color = 'orangered'

    for i, residue in enumerate(Cu_cite_type1):
            plt.axhline(y=residue, color=Cu_cite_color, linewidth=1.7)

    for i, residue in enumerate(Cu_cite_type2):
            plt.axhline(y=residue, color=Cu_cite_color, linestyle='--', linewidth=2)

    cb = plt.colorbar(c, ticks=np.arange(codes_min, codes_max+1), orientation='horizontal', pad=0.15, aspect=50)
    cb.ax.set_xticklabels([reversed_code_mapping[i] for i in range(int(codes_min), int(codes_max)+1)])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel('Residue')
    plt.legend(frameon=False)
    plt.show()
#%%
def open_hbond(filenames):
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
def pca_prep(*universe_discts, end_res):
    frames = []
    CA_noh_res53 = {}
    aligned_CA_noh_res53 = {}
    
    for u_dict in universe_discts: 
        molecule_name, universe = next(iter(u_dict.items()))

        u_CA_noh_res53 = universe.select_atoms(f'protein and name CA and not name H and resid 53:{end_res}')
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
    
    filename = ''.join([name for name in CA_noh_res53.keys()]) + '.pdb'

    with MDAnalysis.Writer(f'/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/{filename}') as W:
        for ts in multi_pdb.trajectory:
            W.write(multi_pdb)
    
    print(frames)
    return multi_pdb, frames, CA_noh_res53

#%%
def pca_analysis(multi_pdb, n_components, monomer_frames=None, chainA_frames=None,
                 chainB_frames=None, chainC_frames=None, elbow_plot=False, show_sequence=False):
    os.chdir('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC')
    M = bb.Molecule()
    M.import_pdb(multi_pdb)
    #how to select residue in Biobox?
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
            # cbar.ax.xaxis.set_label_position('top')
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
#%%
monomer_dict_domain = load_universe('monomer', '5ue6_newbox.gro', 'final_center.xtc', 
                             alignment='domain', stride_step = 200)
monomer_domain = monomer_dict_domain.get('monomer')
#%%
monomer_dict_all = load_universe('monomer', '5ue6_newbox.gro', 'final_center.xtc',
                            alignment='all', stride_step = 100)
monomer_all = monomer_dict_all.get('monomer')
#%%
trimer_dict_domain = load_universe('trimer', 'ABC_newbox.gro', 'final_cluster.xtc', 
                              alignment='domain', stride_step = 100)
trimer_domain = trimer_dict_domain.get('trimer')

trimer_dict_all = load_universe('trimer', 'ABC_newbox.gro', 'final_cluster.xtc', 
                              alignment='all', stride_step = 101)
trimer_all = trimer_dict_all.get('trimer')
#%%
chainA_dict_domain = load_universe('chainA', 'chainA.gro', 'chainA_cluster.xtc', 
                              alignment='domain', stride_step = 100)
chainA_domain = chainA_dict_domain.get('chainA')

chainA_dict_all = load_universe('chainA', 'chainA.gro', 'chainA_cluster.xtc', 
                              alignment='all', stride_step = 100)
chainA_all = chainA_dict_all.get('chainA')
#%%
chainB_dict_domain = load_universe('chainB', 'chainB.gro', 'chainB_cluster.xtc', 
                              alignment='domain', stride_step = 100)
chainB_domain = chainB_dict_domain.get('chainB')

chainB_dict_all = load_universe('chainB', 'chainB.gro', 'chainB_cluster.xtc', 
                              alignment='all', stride_step = 100)
chainB_all = chainB_dict_all.get('chainB')
#%%
chainC_dict_domain = load_universe('chainC', 'chainC.gro', 'chainC_cluster.xtc',
                              alignment='domain', stride_step = 100)
chainC_domain = chainC_dict_domain.get('chainC')

chainC_dict_all = load_universe('chainC', 'chainC.gro', 'chainC_cluster.xtc',
                              alignment='all', stride_step = 100)
chainC_all = chainC_dict_all.get('chainC')
#%%
rmsd_plot(monomer_dict_all, trimer_dict_all, monomer_dict_domain, trimer_dict_domain)
#%%
rmsd_plot(monomer_dict_domain, chainA_dict_domain, chainB_dict_domain, chainC_dict_domain)
#%%
# for trimer chains : ['darkcyan', 'yellowgreen', 'chocolate', 'purple']
rmsd_plot(monomer_domain, chainA_domain, chainB_domain, chainC_domain,
            colors = ['purple', 'darkcyan', 'yellowgreen', 'chocolate'], start_res=[53,53,53,53], end_res=[349,349,349,349])
#%%
pairwise_rmsd(monomer_dict_all_aligned)
pairwise_rmsd(trimer_dict_all_aligned)
#%%
rmsf_plot(monomer_dict_domain_aligned, chainA_dict_domain_aligned, chainB_dict_domain_aligned, chainC_dict_domain_aligned,
           start_res=53, end_res=349, plot_mean=True)
#%%
rmsf_plot(monomer_dict_domain_aligned,
           start_res=53, end_res=362, plot_mean=False)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/vmd_analysis/mon-200dt-17ms.tml', monomer_dict_all, aggregated=True)
#%%
tml_plot('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/vmd_analysis/chainA-100dt-12ms.tml', chainA_dict_all, aggregated=True)
#%%
pca_prep(monomer_dict_all_aligned, end_res=362)
#%%
pca_prep(chainA_dict_all_aligned, chainB_dict_all_aligned, chainC_dict_all_aligned)
pca_prep(monomer_dict_all_aligned, chainA_dict_all_aligned, chainB_dict_all_aligned, chainC_dict_all_aligned)
#%%
pca_analysis('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/monomer.pdb',
             n_components = 2, monomer_frames=3473, show_sequence=True)
# %%
pca_analysis('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/monomerchainAchainBchainC.pdb', n_components = 2,
             monomer_frames=3473, chainA_frames=1210, chainB_frames=1210, chainC_frames=1210, show_sequence=True)
#%%
pca_analysis('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/chainAchainBchainC.pdb', n_components = 2,
             chainA_frames=1210, chainB_frames=1210, chainC_frames=1210, show_sequence=True)
# %%
cmap = cm.get_cmap('RdPu')
position = 0.6  # Example position (50% through the colormap)
rgba = cmap(position)
rgb = rgba[:3]
print("RGB values at position", position, ":", rgb)
fig, ax = plt.subplots(figsize=(2, 2))
ax.set_facecolor(rgb)
# %%
pca_prep(monomer_dict_domain_aligned, end_res=349)
# %%
pca_prep(chainA_dict_domain_aligned, chainB_dict_domain_aligned, chainC_dict_domain_aligned)
pca_prep(monomer_dict_domain_aligned, chainA_dict_domain_aligned, chainB_dict_domain_aligned, chainC_dict_domain_aligned)
# %%
pca_analysis('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/monomer.pdb',
             n_components = 2, monomer_frames=1737, show_sequence=True)
#%%
pca_analysis('/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC/monomerchainAchainBchainC.pdb', n_components = 2,
             monomer_frames=3473, chainA_frames=1210, chainB_frames=1210, chainC_frames=1210, show_sequence=True)
# %%
