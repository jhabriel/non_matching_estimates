# Install `mdamr` from source using `conda`

This is a set of instructions on how to install `mdamr` (and its dependencies including `porepy`) using conda. A pure `pip` installation should also be feasible with minimal modifications.

## Create your `conda` environment

### Check your conda installation
We start by assuming you have conda installed at your local machine. To make sure that this is the case, you can type

```
conda list
```
A succesfull installation will show the list of packages installed in the base environment.

### Make sure the default channel is up to date
To make sure your conda channel is updated, you can type
```
conda update -n base -c default conda
```

### Create a new conda environment
To avoid any potential conflicts with other packages, we recommend creating a new evironment with `python 3.11` installed:
```
conda create --name mdamr python=3.11
```

### Activate your environment
Now that you have created a new environment, you must activate it
```
conda activate mdamr
```

### Install recommended packages beforehand
Experience has shown that before installing `porepy`, it is wise to install some packages beforehand. If you are in an intel-based machine, you can type
```
conda install git jupyter notebook pypardiso
```
and if that is not the case, you can type:
```
conda install git jupyter notebook scikit-umfpack
```

## Install `quadpy`

Unfortunately, the newest versions of `quadpy` are no longer open source. However, the `conda-forge` channel still keeps a copy of the latest open-source version. We need
to install it via:
```
conda install conda-forge::quadpy==0.16.10
```
Note that this will also install the last open-source versions of `orthopy` and `ndim`.

## Install `porepy`

### Install required packages by `porepy`
We can now clone the PorePy repository and install the required pacakges. Note that we are using the stable version `v1.8.1` for this project:
```
git clone -b 'v.1.8.1' https://github.com/pmgbergen/porepy.git porepy181
cd porepy181
conda install --file requirements.txt
```

 ### Install `porepy` via `pip`
 ```
pip install --user -e .
```

### Check your PorePy installation
To check whether PorePy was correctly installed, you can simply type from the root `porepy181` folder
```
pytest
```
Note that all tests should pass.

## Install `mdmamr`

## Install required packages by mdamr
Finally, we need to install `mdamr`. To do this, go to same level as `porepy181` and clone the development version of `mdamr`:
```
git clone https://github.com/jhabriel/mdamr.git
cd mdamr
conda install --file requirements.txt
```

### Install `mdamr` via `pip`
```
pip install --no-deps --user -e .
```
We have to add the `--no-deps` flag, otherwise, `pip` will not find the required versions of `quadpy`, `ndim`, and `orthopy`. 

### Check you `mdamr` installation
Again, to check whether `mdamr` was correctly installed, you can move to the `mdamr` root folder and type
```
pytest
```

