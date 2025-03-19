import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData
import uncertainties as uc
from uncertainties import unumpy

def nnn(x):
    return x.n
def sss(x):
    return x.s

def linear_model(params, x):
    k, b = params
    return k * x + b

df = pd.read_excel("211.xlsx")

mA1 = unumpy.uarray(df['mA1'].to_numpy(), df['DmA'].to_numpy())
mA2 = unumpy.uarray(df['mA2'].to_numpy(), df['DmA'].to_numpy())
U1 = unumpy.uarray(df['U1'].to_numpy(), df['DU'].to_numpy())
U2 = unumpy.uarray(df['U2'].to_numpy(), df['DU'].to_numpy())
mU1 = unumpy.uarray(df['mU1'].to_numpy(), df['DmU'].to_numpy())
mU2 = unumpy.uarray(df['mU2'].to_numpy(), df['DmU'].to_numpy())
qlm1 = unumpy.uarray(df['qlm1'].to_numpy(), df['Dqlm'].to_numpy())
qlm2 = unumpy.uarray(df['qlm2'].to_numpy(), df['Dqlm'].to_numpy())

b = uc.ufloat(40.7, 0.05) / 1000

N1 = U1 * mA1 / 1000
N2 = U2 * mA2 / 1000
delT1 = mU1 / b
delT2 = mU2 / b

x = np.array(list(map(nnn, delT1)))
y = np.array(list(map(nnn, N1)))
xer = np.array(list(map(sss, delT1)))
yer = np.array(list(map(sss, N1)))

data = RealData(x, y, sx = xer, sy = yer)
model = Model(linear_model)
odr = ODR(data, model, beta0=[1.0, 0.0])
output = odr.run()

k = output.beta[0]
b = output.beta[1]
k_err = output.sd_beta[0]
b_err = output.sd_beta[1]

plt.errorbar(x, y,xerr = xer, yerr = yer, fmt = 'o')
plt.show()