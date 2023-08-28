#!/bin/bash
readonly packages=(
  python 
  plotly 
  pandas 
  numpy 
  # additional modules are shown which are part of the scicomp env
  natsort 
  openpyxl
  xlrd
  matplotlib 
  pyarrow
  feather-format
  bokeh
  seaborn 
  ipython 
  zstandard
  tqdm
  mpmath
  h5py
  networkx
  dask
  scikit-learn
  # for integration with jupyter
  nodejs
  ipykernel 
  voila 
  ipywidgets=7 
)

# note assumption that mamba is in env
mamba create -p /packages/envs/scicomp -c conda-forge "${packages[@]}"
