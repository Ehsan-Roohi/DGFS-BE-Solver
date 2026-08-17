### 2D/3V Flux-reconstructed Discontinuous Galerkin Fast Spectral (DGFS)
is an open-source software code for solving single/multi-species Boltzmann equation and related kinetic models for gas flows. It is designed for CUDA-enabled GPUs 
and applies the high-order flux-reconstructed discontinuous Galerkin schemes. The primary codebase for the DGFS has been originally developed by Dr. Shashank Jaiswall as part of his PhD dissertation research supported under the NSF award "CDS&E: Deterministic Evaluation of Kinetic Boltzmann equation with Spectral H/p/v Accuracy".  

The Boltzmann equation describes the evolution of the molecular velocity distribution function $f=f(t,\boldsymbol{x},\boldsymbol{v})$.

$$
\partial_t f + \boldsymbol{v}\cdot\nabla_{\boldsymbol{x}} f
= Q(f,f_*), \quad
t>0,\; \boldsymbol{x}\in\Omega\subset\mathbb{R}^3,\;
\boldsymbol{v}\in\mathbb{R}^3 .
$$


The collision integral calculations use two methodologies: 
* the explicit fast spectral schemes for full Boltzmann **[Gamba 2017, Jaiswal 2019a, Jaiswal 2019b]**. The method applies to general collision kernels, and the results can be "directly" compared with the DSMC without need of any recalibration or parametric fitting.  
* the implicit schemes for linear kinetic models  **[Dimarco 2013, Dimarco 2017]**

The time integration is performed via: 
* 1st/2nd order explicit Strong Stability Preserving (SSP) Runge Kutta (RK) schemes
* 1st/2nd/3rd order implicit-explicit Ascher-Ruuth-Spiteri (ARS) and backward-difference (BDF) schemes  

The overall schemes are simple from mathematical and implementation perspective; highly accurate in both physical and velocity spaces as well as time; robust, i.e. applicable for general geometry and spatial mesh; exhibits nearly linear parallel scaling; and directly applies to general collision kernels needed for high fidelity flow modelling.  

### Requirements:

System Requirements:
	•	Linux OS (Ubuntu, CentOS, etc.)
	•	GCC ≥ 6.3.0
	•	OpenMPI ≥ 3.1.6 (with CUDA support recommended)
	•	CUDA Toolkit ≥ 11.0
	•	Python 3.8+
	•	GMSH 2.15.0 (for mesh generation)
	•	METIS (for mesh partitioning)

Python Dependencies (install via pip):
	•	numpy < 1.24
	•	h5py
	•	mpi4py
	•	pycuda
	•	gimmick==2.3 (must be pinned to 2.3 or lower to avoid compatibility issues)


## DGFS-BE Solver Installation Instructions

The following instructions describe how to install the DGFS-BE Solver on a Linux system using Python virtual environments, GMSH, and METIS.

---

### 1. Prepare Directory Structure

```bash
mkdir -p ~/sourceCodes/DGFS/gmsh
cd ~/sourceCodes/DGFS
```

Clone the solver and test case repositories:

```bash
git clone https://github.itap.purdue.edu/DGFSproj/DGFS-BE-Solver.git
git clone https://github.itap.purdue.edu/DGFSproj/DGFS-BE-TestCases.git
```

Download the following required archives:

- **METIS 5.1.0**:  
  https://sourceforge.net/projects/openfoam-extend/files/foam-extend-3.0/ThirdParty/metis-5.1.0.tar.gz/download  
  → place in `~/sourceCodes/DGFS/`

- **GMSH 2.15.0**:  
  https://gmsh.info/bin/Linux/gmsh-2.15.0-Linux64.tgz  
  → place in `~/sourceCodes/DGFS/gmsh/`

---

### 2. Load Required Modules

Purge existing modules and load verified versions:

```bash
module purge
module load gcc/11.4.1
module load openmpi/5.0.5
module load cuda/12.1.0
```

> These versions are verified compatible as of September 2025.

---

### 3. Set Up Python Environment

Create and activate the Python environment inside the DGFS folder:

```bash
cd ~/sourceCodes/DGFS
python3 -m venv $PWD/dgfs_env
source ~/sourceCodes/DGFS/dgfs_env/bin/activate
```

---

### 4. Install Python Dependencies

Install core Python packages:

```bash
pip install "numpy<1.24"
pip install h5py
pip install mpi4py
pip install pycuda
```

Manually install a compatible version of `gimmick` (2.3 or lower; version 1 was verified with provided instruction):

```bash
pip install --upgrade pip
pip install gimmick==1.0 --no-deps
```

---

### 5. Install DGFS Solver

From inside the solver directory:

```bash
cd ~/sourceCodes/DGFS/DGFS-BE-Solver
python setup.py install
cd ..
```

---

### 6. Install METIS

Extract, configure, and build METIS:

```bash
tar -xvf metis-5.1.0.tar
cd metis-5.1.0
nano Makefile  # change: shared = yes, then save and exit
make config prefix=$PWD/build
make install
export FRFS_METIS_LIBRARY_PATH=$PWD/build/lib/libmetis.so
cd ..
```

---

### 7. Install and Link GMSH

Unpack the GMSH archive:

```bash
cd ~/sourceCodes/DGFS/gmsh
tar -xvf gmsh-2.15.0-Linux64.tar
```

Add an alias to your shell config:

```bash
nano ~/.bashrc
```

Add this line at the end:

```bash
alias gmsh=~/sourceCodes/DGFS/gmsh/gmsh-2.15.0-Linux/bin/gmsh
```
Make the binary executable:

```bash
chmod +x ~/sourceCodes/DGFS/gmsh/gmsh-2.15.0-Linux/bin/gmsh
```

Then reload:

```bash
source ~/.bashrc
```

---

### 8. Before Running Any Case

Each time before you run DGFS, activate the environment and export the METIS path:

```bash
export FRFS_METIS_LIBRARY_PATH=~/sourceCodes/DGFS/metis-5.1.0/build/lib/libmetis.so
source ~/sourceCodes/DGFS/dgfs_env/bin/activate
module load gcc/11.4.1
module load openmpi/5.0.5
module load cuda/12.1.0
```


### References:
* **[Gamba 2017]** Gamba, I. M., Haack, J. R., Hauck, C. D., & Hu, J. (2017). 
  *A fast spectral method for the Boltzmann collision operator with general collision kernels.* SIAM Journal on Scientific Computing, 39(4), B658-B674.
* **[Dimarco 2013]** Dimarco, G., & Pareschi, L. (2013). 
  *Asymptotic preserving implicit-explicit Runge--Kutta methods for nonlinear kinetic equations.* SIAM Journal on Numerical Analysis, 51(2), 1064-1087.
* **[Dimarco 2017]** Dimarco, G., & Pareschi, L. (2017). 
  *Implicit-explicit linear multistep methods for stiff kinetic equations.* SIAM Journal on Numerical Analysis, 55(2), 664-690.
* **[Jaiswal 2019a]** Jaiswal, S., Alexeenko, A. A., and Hu, J. (2019)
  *A discontinuous Galerkin fast spectral method for the full Boltzmann equation with general collision kernels.* Journal of Computational Physics 378: 178-208. https://doi.org/10.1016/j.jcp.2018.11.001
* **[Jaiswal 2019b]** Jaiswal, S., Alexeenko, A. A., and Hu, J. (2019)
  *A discontinuous Galerkin fast spectral method for the multi-species full Boltzmann equation.* Computer Methods in Applied Mechanics and Engineering 352: 56-84. https://doi.org/10.1016/j.cma.2019.04.015
* **[Jaiswal 2019d]** Jaiswal, S., Pikus, A., Strongrich A., Sebastiao I. B., Hu, J., and Alexeenko, A. A. (2019)
  *Quantification of thermally-driven flows in microsystems using Boltzmann equation in deterministic and stochastic context.* Physics of Fluids 31(8): 082002. https://doi.org/10.1063/1.5108665

Please cite this work as:

Vorozhbit, E.; Morton, B.; Adhikari, N.; Alexeenko, A. A.; Hu, J. DGFS-BE Solver: An open-source Discontinuous Galerkin Fast Spectral Solver for the full Boltzmann equation. SoftwareX, 34 (2026), 102544. https://doi.org/10.1016/j.softx.2026.102544.

### Reproducible paper case

The exact eight-element helium normal-shock case from Figure 14(b) of
Jaiswal, Alexeenko, and Hu (JCP 378, 2019) is available in
[`cases/jcp2019_fig14b_normal_shock`](cases/jcp2019_fig14b_normal_shock).
It includes a Unity bootstrap, a paper-parameter audit, automatic checksums,
and raw-DG versus exact cell-average post-processing.

### License:
*DGFS* is released as GNU GPLv2 open-source software. 
This code has been derived from "PyFR". Please see licenses folder for restrictions.

