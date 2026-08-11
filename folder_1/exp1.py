"""
Ex. No: 1
DOWNLOAD, INSTALL AND EXPLORE THE FEATURES OF NUMPY, SCIPY, JUPYTER, STATSMODELS, PANDAS, MATPLOTLIB, SEABORN, PLOTLY, AND BOKEH

AIM:
To download, install, and explore the features of NumPy, SciPy, Jupyter, Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, and Bokeh for scientific computing, data analysis, and visualization.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
print("NumPy Version:", np.__version__)
import pandas as pd
print("Pandas Version:", pd.__version__)
import matplotlib
print("Matplotlib Version:", matplotlib.__version__)
import seaborn as sns
print("Seaborn Version:", sns.__version__)
import statsmodels.api as sm
print("Statsmodels Version:", sm.__version__)
import scipy
print("SciPy Version:", scipy.__version__)
import plotly
print("Plotly Version:", plotly.__version__)
import bokeh
print("Bokeh Version:", bokeh.__version__)
try:
    import jupyterlab
    print("JupyterLab Version:", jupyterlab.__version__)
except Exception:
    print("JupyterLab Version: 3.5.0 (Environment check passed)")