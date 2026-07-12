# QM9
QM9 is a widely used quantum chemistry dataset containing computed structures and properties for small organic molecules. It has become one of the standard benchmarks for developing and evaluating machine learning models that predict molecular properties, particularly graph neural networks and other molecular representation methods. 

## What it contains
The original QM9 dataset comprises approximately 134,000 stable organic molecules derived from the GDB-17 chemical space. After removing molecules with problematic calculations, many software packages distribute a cleaned version containing 130,831 molecules. The molecules contain only the elements:

* Carbon (C)
* Hydrogen (H)
* Oxygen (O)
* Nitrogen (N)
* Fluorine (F)

Each molecule has up to nine heavy (non-hydrogen) atoms and includes its optimized 3D geometry together with numerous quantum-chemical properties computed using density functional theory (DFT). 

## Available properties
QM9 provides multiple target properties for each molecule, including:
* Dipole moment
* Polarizability
* HOMO and LUMO orbital energies
* HOMO–LUMO gap
* Electronic spatial extent
* Zero-point vibrational energy
* Internal energy
* Enthalpy
* Gibbs free energy
* Heat capacity
* Atomization energies

These properties make QM9 a multi-target regression benchmark rather than a simple classification dataset. 

Source: https://kumo.ai/pyg/datasets/qm9

## Why it is important
QM9 became the de facto benchmark for molecular machine learning because it combines:
* A relatively large number of molecules
* Consistent quantum-chemical calculations
* Rich molecular property annotations
* High-quality optimized molecular geometries

It has been used to benchmark message-passing neural networks, equivariant neural networks, transformer-based molecular models, and many other graph learning architectures. Results reported on QM9 are often used to compare new molecular ML methods against prior work. 

Source: https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2024-w3ld0-v2

## Limitations
Although extremely influential, QM9 has several important limitations:
* It contains only small organic molecules.
* It includes only five chemical elements (H, C, N, O, and F).
* The calculations are performed at a single level of quantum theory, so they do not represent experimental measurements.
* It does not adequately cover larger drug-like molecules, inorganic chemistry, transition metals, or chemical reactions.

Because of these limitations, newer datasets have expanded QM9 with additional elements, larger molecules, excited-state properties, higher-accuracy calculations, or solvent effects.


[Molecular Quantum Chemical Data Sets and Databases for Machine Learning Potentials, August 22, 2024](https://arxiv.org/abs/2408.12058)

[Hessian QM9: A quantum chemistry database of molecular Hessians in implicit solvents, August 15, 2024](https://arxiv.org/abs/2408.08006)

[Accurate GW frontier orbital energies of 134 kilo molecules, March 15, 2023](https://arxiv.org/abs/2303.08708)

[Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models, June 22, 2019](https://arxiv.org/abs/1906.09427)

[Hessian QM9: A quantum chemistry database of molecular ..., Nature](https://www.nature.com/articles/s41597-024-04361-2)

[MultiXC-QM9: Large dataset of molecular and reaction ..., Nature](https://www.nature.com/articles/s41597-023-02690-2)

[Quantum Chemistry Dataset with Ground- and Excited ..., by Y Zhu · 2024](https://www.nature.com/articles/s41597-024-03788-x)

[qm9 | TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/qm9)

["Quantum chemistry structures and properties of 134 kilo molecules.", Kaggle](https://www.kaggle.com/datasets/mariovozza5/qm9-molecules)

[Quantum Machine 9, aka QM9, Kaggle](https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9)

[Machine Learning Prediction of Nine Molecular Properties ...](https://www.researchgate.net/publication/346029441_Machine_Learning_Prediction_of_Nine_Molecular_Properties_Based_on_the_SMILES_Representation_of_the_QM9_Quantum-Chemistry_Dataset)


[Results of Quantum Chemical and Machine Learning Computations for Molecules in the QM9 Database](https://purl.stanford.edu/kf921gd3855)


[QM9, Description: QM9 dataset is an enumeration of around 134k stable organic molecules with up to 9 heavy atoms (carbon, oxygen, nitrogen and fluorine)](https://graphgt.github.io/molecule.html)

[QM7 dataset](https://quantum-machine.org/datasets)

[OPEN QUANTUM CHEMISTRY PROPERTY DATABASE ... by W Liu](https://openreview.net/pdf?id=o6aUi3ukdd)

[Molecular Quantum Chemical Data Sets and Databases for ... by A Ullah · 2024](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2024-w3ld0-v2) 

[Quantum Chemistry Dataset with Ground- and Excited-state ... by Y Zhu · 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362161/)

[Machine Learning Prediction of Nine Molecular Properties ... Nov 11, 2020](https://pubs.acs.org/doi/10.1021/acs.jpca.0c05969)

[QM9 Dataset: Quantum Chemistry Molecular Properties](https://kumo.ai/pyg/datasets/qm9/)

[QM9: Quantum Chemistry Properties of 134k Molecules, 11 Apr 2026](https://hunterheidenreich.com/notes/chemistry/datasets/qm9/) 

[torch_geometric.datasets.QM9 - PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html)

[Quantum chemistry structures and properties of 134 kilo molecules](https://arxiv.org/html/2511.21747v1)

[QuantumChem-200K: A Large-Scale Open Organic ..., 23 Nov 2025](https://arxiv.org/html/2511.21747v1)

[Wiki: Q-Chem](https://en.wikipedia.org/wiki/Q-Chem)

[Wiki: Quantum computational chemistry](https://en.wikipedia.org/wiki/Quantum_computational_chemistry)

[Quantum chemistry composite methods](https://en.wikipedia.org/wiki/Quantum_chemistry_composite_methods)

[Jose Luis Mendoza-Cortes](https://en.wikipedia.org/wiki/Jose_Luis_Mendoza-Cortes)