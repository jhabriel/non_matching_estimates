# Install `mdnme` from source

This is a set of instructions on how to install `mdnme` (and its dependencies including `porepy`) using `conda`. A pure `pip` installation should also be feasible with minimal modifications.

## Create your `conda` environment

### Check your conda installation
We start by assuming you have conda installed at your local machine. To make sure that this is the case, you can type

```
conda list
```
A succesful installation will show the list of packages installed in the base environment.

### Make sure the default channel is up to date
To make sure your conda channel is updated, you can type
```
conda update -n base -c default conda
```

### Create a new conda environment
To avoid any potential conflicts with other packages, we recommend creating a new environment with `python 3.11` installed:
```
conda create --name mdnme python=3.11
```

### Activate your environment
Now that you have created a new environment, you must activate it
```
conda activate mdnme
```

### Install recommended packages beforehand
Experience has shown that before installing `porepy`, it is wise to install some packages beforehand. If you are in an intel-based machine, you can type
```
conda install git jupyter notebook pypardiso
```
and if that is not the case (e.g., for Apple-Silicon Macs), you can type:
```
conda install git jupyter notebook scikit-umfpack
```

## Install `quadpy`

Unfortunately, the newest versions of `quadpy` are no longer open source. However, the `conda-forge` channel still keeps a copy of the latest open-source version. We need to install it via:
```
conda install conda-forge::quadpy==0.16.10
```
Note that this will also install the latest open-source versions of `orthopy` and `ndim`, which are dependencies of `quadpy`.

## Install `porepy`

### Install required packages by `porepy`
We can now clone the PorePy repository and install the required packages. Note that we require the commit b5202e8 to be installed for this project:
```
git clone https://github.com/pmgbergen/porepy.git porepy
cd porepy
git checkout b5202e81fc6a2c203fae3d4be066dd866978f882
```

### Install `porepy` via `pip`
```
pip install -e .
```

### Check your PorePy installation
To check whether PorePy was correctly installed, from the root `porepy` folder type
```
pytest
```
All tests should pass or something went wrong with your installation.

## Install `mdmne`

## Install required packages by mdnme
Finally, we need to install `mdnme`. To do this, go to same level as `porepy` and clone the development version of `mdmne`:
```
cd ..
git clone https://github.com/jhabriel/non_matching_estimates.git mdnme
cd mdnme
```

### Install `mdnme` via `pip`
```
pip install --no-deps -e .
```

### Check your `mdnme` installation
To check whether `mdnme` was correctly installed, you can move to the `mdnme` root folder and type
```
pytest
```
Again, all tests should pass.
