import numpy as np
import matplotlib.pyplot as plt

dets = np.load('./output/dets.npy')
fas = np.load('./output/fas.npy')
ntgt = np.load('./output/ntgt.npy')[0]
nframes = np.load('./output/nframes.npy')[0]
print(ntgt,nframes,len(dets),len(fas))

maxv=max(max(fas), max(dets))
minv=min(min(fas),min(dets))
step=(maxv-minv)/1000;
print(maxv,minv)
pds=[]
fars=[]
t = minv
while t < maxv:

    x = np.where(dets > t)
    pd = x[0].shape/ntgt
    pds.append(pd)

    y = np.where(fas>t)
    far = y[0].shape/(nframes * 3.4 * 2.6)
    fars.append(far)
    t += step

print(len(pds),len(fars))
plt.plot(fars,pds)
plt.savefig("./output/" + "roc_curve_TCRNet1_example")
plt.show()