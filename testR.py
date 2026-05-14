import pandas as pd
import os
os.environ["R_HOME"] = "C:/PROGRA~1/R/R-4.6.0"
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

data = pd.read_parquet('supplies_data.parquet')
with localconverter(ro.default_converter + pandas2ri.converter):
    r_dataframe = ro.conversion.py2rpy(data)

r.source('cositaR.R')
test_func_R = ro.globalenv['test']

resultado = test_func_R(r_dataframe)
print(resultado)