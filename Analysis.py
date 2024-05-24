#%%
import os
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from MDAnalysis.analysis import pca, diffusionmap, rms, align
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


class SingleAnalysis:
    def __init__(self, directory, gro_file, xtc_file):
        self.directory = directory
        self.gro_file = gro_file
        self.xtc_file = xtc_file
        self.aligned_xtc_file = f"{os.path.splitext(xtc_file)[0]}_aligned.xtc"
        self.universe = None
        self.reference = None
    
    def change_directory(self):
        try:
            os.chdir(self.directory)
            print(f"Successfully changed the working directory to {self.directory}")
        except Exception as e:
            print(f"Error occurred while changing the working directory: {e}")

    def load_universe(self):
        if os.path.exists(self.aligned_xtc_file):
            print('Aligned trajectory found. Loading the aligned trajectory file')
            traj_to_load = self.aligned_xtc_file
        else:
            print('Aligned trajectory file does not exist. Proceeding with alignment')
            self.universe = mda.Universe(self.gro_file, self.xtc_file)
            self.reference = mda.Universe(self.gro_file, self.xtc_file)
            print('Aligning the trajectory. This may take a while')
            aligner = align.AlignTraj(self.universe, self.reference, select = 'protein', filename = f"{os.path.splitext(self.xtc_file)[0]}_aligned.xtc").run()
            print('trajectory aligned and saved')
            traj_to_load = aligner

        self.universe = mda.Universe(self.gro_file, traj_to_load)
        self.reference = mda.Universe(self.gro_file, traj_to_load)

        print("Aligned universe loaded.")
    
    def write_sliced_traj(self, slice_step):
        output_filename = f"{os.path.splitext(self.xtc_file)[0]}_sliced_{slice_step}.xtc"
        with XTCWriter(output_filename, n_atoms = self.universe.atoms.n_atoms) as writer:
            for ts in self.universe.trajectory[::slice_step]:
                writer.write(self.universe.atoms)  

    
    def calculate_rmsd(self):
        self.universe.trajectory[-1]
        self.reference.trajectory[0]
        self.u_ca = self.universe.select_atoms("name CA")
        self.ref_ca = self.reference.select_atoms("name CA")
        self.aligned_rmsd = rms.rmsd(self.u_ca.positions, self.ref_ca.positions, superposition=False)
        print(f"Aligned RMSD: {self.aligned_rmsd:.2f}")

    def plot_rmsd(self):
        self.R = rms.RMSD(self.universe, self.reference, select = "backbone").run()
        self.df = pd.DataFrame(self.R.rmsd, columns = ['Frame', r'Time ($\mu$s)', 'RMSD'])
        
        self.ax = self.df.plot(x = r'Time ($\mu$s)', y = 'RMSD', kind = 'line')
        self.ax.set_ylabel(r'RMSD ($\AA$)')
        return self.ax

    def plot_rmsf(self, start_res=None, end_res=None):
        backbone = self.universe.select_atoms("backbone")
        selected_resids = backbone.resids
        self.R = rms.RMSF(backbone).run()
        rmsf = self.R.results.rmsf

        if start_res is not None and end_res is not None:
            indices = (backbone.resids >= start_res) & (backbone.resids <= end_res)
            selected_resids = backbone.resids[indices]
            rmsf = self.R.results.rmsf[indices]

        fig, ax = plt.subplots()
        ax.plot(selected_resids, rmsf)
        plt.xlabel('Residue number')
        plt.ylabel('RMSF ($\AA$)')
        plt.show()

    def rmsd_matrix(self, slice_step):
        sliced_xtc_file = f"{os.path.splitext(self.xtc_file)[0]}_sliced_{slice_step}.xtc"
        with XTCWriter(sliced_xtc_file, n_atoms = self.universe.atoms.n_atoms) as writer:
            for ts in self.universe.trajectory[::slice_step]:
                writer.write(self.universe.atoms)  

        self.universe = mda.Universe(self.gro_file, sliced_xtc_file)
        matrix = diffusionmap.DistanceMatrix(self.universe, select='name CA').run()
        
        frame_num = len(self.universe.trajectory)
        total_time = frame_num * slice_step * 20 / 1000000

        plt.imshow(matrix.dist_matrix, cmap='viridis', extent=[0,total_time, 0,total_time])
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel(r'Time ($\mu$s)')
        plt.colorbar(label=r'RMSD ($\AA$)')
        plt.show()
    
    def plot_timeline(tml_file, aggregated = False):
        residues = []
        time = []
        codes = []
        with open(tml_file, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.split() 
                if len(parts) < 3:  
                    continue
        residues.append(int(parts[0])) 
        time.append(0.01*int(parts[-2]))  
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

        viridis = plt.cm.get_cmap('viridis', len(code_mapping))  
        colors = viridis(np.linspace(0, 1, len(code_mapping))) 
        cmap = ListedColormap(colors)
        codes_min, codes_max = pivoted_df.min().min(), pivoted_df.max().max()
        boundaries = np.arange(codes_min-0.5, codes_max+1.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)
        
        plt.figure(figsize=(12, 6))
        c = plt.pcolormesh(pivoted_df.columns, pivoted_df.index, pivoted_df, cmap=cmap, norm=norm, shading='auto')

        cb = plt.colorbar(c, ticks=np.arange(codes_min, codes_max+1))
        cb.ax.set_yticklabels([reversed_code_mapping[i] for i in range(int(codes_min), int(codes_max)+1)])

        plt.xlabel('Time (ns)')
        plt.ylabel('Residue')
        plt.show()
#%%
class MultiAnalysis:
    def __init__(self, *configs):
        self.configs = configs
        self.universes = {}
        self.directories = {
            'monomer':'/home/pghw87/Documents/md-sim/5ue6/monomer/monomer',
            'trimer':'/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'
        }
        for config in configs:
            molecule_type, file_name = config
            self.setup_analysis(molecule_type, file_name)
    
    def setup_analysis(self, molecule_type, file_name):
        directory = self.directories.get(molecule_type)
        topology_file = f"{file_name}.gro"
        trajectory_file = f"{file_name}.xtc"
        aligned_trj_file = f"{file_name}_aligned.xtc"

        try:
            os.chdir(directory)
            print(f"changed directory to {directory}")
        except Exception as e:
            print(f"error chaging directory as {e}")
            return
        
        self.load_universe(molecule_type, topology_file, trajectory_file, aligned_trj_file)

    def load_universe(self, molecule_type, topology_file, trajectory_file, aligned_trj_file):
        if os.path.exists(aligned_trj_file):
            print(f"Aligned trajectory for {molecule_type} found. Loading...")
            universe = mda.Universe(topology_file, aligned_trj_file)
        else:
            print(f"No aligned trajectory file found for {molecule_type}. Aligning...")
            universe = mda.Universe(topology_file, trajectory_file)
            reference = mda.Universe(topology_file, trajectory_file)
            align.AlignTraj(universe, reference, select = 'protein', filename = aligned_trj_file ).run()
            universe = mda.Universe(topology_file, aligned_trj_file)

        self.universes[molecule_type] = universe
        print(f"{molecule_type} universe loaded.")


# %%
