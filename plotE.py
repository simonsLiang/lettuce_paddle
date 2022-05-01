import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default='')
parser.add_argument("--savename", type=str, default='./')
args = parser.parse_args()
data = np.load(args.filename)
ek = data[:, 2]
el = data[:, 1]
Es2 = [ek[i] for i in range(len(ek)) if i % 6 == 0]
el2 = [el[i] for i in range(len(el)) if i % 6 == 0]
nres = [-(Es2[i + 1] - Es2[i]) for i in range(len(Es2) - 1)]
plt.xlim(0,11)
plt.ylim(0,0.017)
plt.plot(el2[:-1], nres)
plt.savefig(args.savename)
# plt.show()