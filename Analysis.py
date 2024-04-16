#%%
import os
import MDAnalysis as mda
from MDAnalysis.analysis import pca, diffusionmap, rms, align

class Analysis:
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
            aligner = align.AlignTraj(self.universe, self.reference, select = 'protein', filename = self.aligned_xtc_file).run()
            print('trajectory aligned and saved')
            traj_to_load = self.aligned_xtc_file

        self.universe = mda.Universe(self.gro_file, traj_to_load)
        self.reference = mda.Universe(self.gro_file, traj_to_load)

        print("Aligned universe loaded.")
    # %%
