import numpy as np
import math
import segyio
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

#
f3_horizon1_file = "h1.txt"
f3_horizon2_file = "h2.txt"
f3_horizon3_file = "h3.txt"
f3_horizon4_file = "h4.txt"
f3_horizon5_file = "h5.txt"

#
f3_horizon1 = []
f3_horizon2 = []
f3_horizon3 = []
f3_horizon4 = []
f3_horizon5 = []

#
with open(f3_horizon1_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    for line in file:
        #
        values = line.strip().split()
        #
        f3_horizon1.append(float(values[5]))

#
with open(f3_horizon2_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    for line in file:
        #
        values = line.strip().split()
        #
        f3_horizon2.append(float(values[5]))

#
with open(f3_horizon3_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    for line in file:
        #
        values = line.strip().split()
        #
        f3_horizon3.append(float(values[5]))

#
with open(f3_horizon4_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    for line in file:
        #
        values = line.strip().split()
        #
        f3_horizon4.append(float(values[5]))

#
with open(f3_horizon5_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    for line in file:
        #
        values = line.strip().split()
        #
        f3_horizon5.append(float(values[5]))

f3_horizon1 = np.array(f3_horizon1).reshape(1,224)
f3_horizon2 = np.array(f3_horizon2).reshape(1,224)
f3_horizon3 = np.array(f3_horizon3).reshape(1,224)
f3_horizon4 = np.array(f3_horizon4).reshape(1,224)
f3_horizon5 = np.array(f3_horizon5).reshape(1,224)

#
f3_horizon = np.zeros(shape=(1,224,224)).astype(np.float32())

for i in range(len(f3_horizon1)):
    for j in range(len(f3_horizon1[i])):
        time1 = f3_horizon1[i][j]
        time2 = f3_horizon2[i][j]
        time3 = f3_horizon3[i][j]
        time4 = f3_horizon4[i][j]
        time5 = f3_horizon5[i][j]
        f3_horizon[i,j,int(math.floor(time1))] = float(1)
        f3_horizon[i,j,int(math.floor(time2))] = float(1)
        f3_horizon[i,j,int(math.floor(time3))] = float(1)
        f3_horizon[i,j,int(math.floor(time4))] = float(1)
        f3_horizon[i,j,int(math.floor(time5))] = float(1)

print(f3_horizon.shape)

f3_fault1 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault1_file = "f1.txt"
#
with open(f3_fault1_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault1[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault1[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault1[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1


f3_fault2 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault2_file = "f2.txt"

with open(f3_fault2_file, 'r', encoding='utf-8') as file:

    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault2[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault2[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault2[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1


f3_fault3 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault3_file = "f3.txt"
#
with open(f3_fault3_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault3[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault3[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault3[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1


f3_fault4 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault4_file = "f4.txt"
#
with open(f3_fault4_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault4[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault4[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault4[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1


f3_fault5 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault5_file = "f5.txt"
#
with open(f3_fault5_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault5[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault5[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault5[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1


f3_fault6 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault6_file = "f6.txt"
#
with open(f3_fault6_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault6[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault6[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault6[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1


f3_fault7 = np.zeros(shape=(1,224,224)).astype(np.float32())
#
f3_fault7_file = "f7.txt"
#
with open(f3_fault7_file, 'r', encoding='utf-8') as file:
    #
    for _ in range(6):
        next(file)
    i = 1
    for line in file:
        if(i==1):
            #
            values = line.strip().split()
            #
            f3_fault7[:,int(values[1])-300,:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
        else:
            #
            values = line.strip().split()
            #
            f3_fault7[:,int(values[1])-300,int(float(laster_values[4]))-4:int(float(values[4]))-4] = 1
            laster_values = values
            i = i + 1
    f3_fault7[:, int(values[1]) - 300, int(float(laster_values[4])) - 4:] = 1

f3_fault = f3_fault1 + f3_fault2 + f3_fault3 + f3_fault4 + f3_fault5 + f3_fault6 + f3_fault7
f3_fault = np.where((f3_fault == 2), 1, f3_fault)

# f3_fault = segyio.cube("F3_Horizon_and_fault/fpx.sgy")
# f3_fault = np.where((f3_fault <= 0.5), 0, 1)

f3_label = f3_horizon + f3_fault
f3_label = np.where((f3_label == 2), 1, f3_label)

plt.figure(figsize=(10, 6))
plt.imshow(f3_label[:,:,:].T, cmap="seismic")
plt.xlabel("Trace Number")
plt.ylabel("Time (ms)")
plt.show()

f3_label.tofile('F3_dataset/' + 'label.dat')