#%%
import os
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
from MDAnalysis.analysis import pca, diffusionmap, rms, align
import pandas as pd
import matplotlib.pyplot as plt


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
        total_time = frame_num * slice_step * 20

        plt.imshow(matrix.dist_matrix, cmap='viridis', extent=[0,total_time, 0,total_time])
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel(r'Time ($\mu$s)')
        plt.colorbar(label=r'RMSD ($\AA$)')
        plt.show()

class MultiAnalysis:
    def __init__(self, monomer=None, chainA, chainB=None, chainC=None):
        self.trimer_path = '/home/pghw87/Documents/md-sim/5ue6/trimer/ABC/ABC'
        
        if monomer is not None:
            self.monomer_path = '/home/pghw87/Documents/md-sim/5ue6/monomer/monomer/'
            self.mon_gro_file = 'data/self.monomer'
            self.mon_xtc_file = 'self.monomer'
        if chainA is not None:
            
            self.A_gro_file = 'data/self.chainA'
            self.A_xtc_file = 'self.chainA'
            




