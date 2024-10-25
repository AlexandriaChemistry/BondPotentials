BondPotentials
==============

Scripts and data to derive parameters for various potentials for diatomic compounds. Some of the scripts sport a -h option to get some help and information.

+ curve_fit.py will perform curve fitting of 28 potentials to 71 diatomic energy curves from either MP2 or CCSD(T) with a basis set of aug-cc-pvtz. Alternatively, a fit can be done to experimental data to try and reproduce the results from a paper by Royappa et al. https://doi.org/10.1016/j.molstruc.2005.11.008
Output is produced in a csv file, a pdf with plots of the fits (if requested) etc. Includes -h flag for help.

+ frequencies.py will estimate the ground state vibrational frequency of the quantum chemistry calculations or the fits and produce plots and tables. Includes -h flag for help.

+ analyse_z.py will summarize the results of curve fitting by averaging over potentials and produce latex tables used in the submitted manuscript.

+ data_graphs.py will plot potentials from quantum chemistry and, where available, experimental data.

+ make_fitting_bounds.py will read the json files containing the fitted parameters and generate a new potentials.json that can be used to limit the search in fitting to a minimum and a maximum.

+ diatomics.py is a utility code to read data files.

+ potentials.py is the code to compute potentials.

Data for diatomic molecules were taken from [The Diatomic Molecular Spectroscopy Database](https://rios.mp.fhi.mpg.de). Please note though, that some errors were detected and some omissions were found that we have addressed in an additional file in the data directory.

A manuscript about this the code and results from this repository, entitled *Quantitative Evaluation of Anharmonic Bond Potentials for Molecular Simulations*, by Paul J. van Maaren and David van der Spoel has been submitted for publication.
