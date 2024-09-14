# Create a conda env for single cell analysis with python 3.8:
conda create -n singleCell python=3.8 h5py openpyxl numpy scipy pandas matplotlib seaborn scikit-learn jupyterlab scanpy python-igraph leidenalg
# Activcate:
conda activate singleCell
# Use pip to install the adap package for transfer learning algorithms:
pip install adap
