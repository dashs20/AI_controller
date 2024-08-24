import matplotlib.pyplot as plt
import pickle
import numpy as np

num = 1

with open(f"prev_runs/time_hists{num}.pkl", 'rb') as picklefile:
    time_hists = pickle.load(picklefile)

with open(f"prev_runs/tvec{num}.pkl", 'rb') as picklefile:
    tvec = pickle.load(picklefile)

with open(f"prev_runs/force_hists{num}.pkl", 'rb') as picklefile:
    force_hists = pickle.load(picklefile)

# plot results
for i in range(len(force_hists)):
    plt.plot(tvec,time_hists[i][:,0])

plt.figure()
for i in range(len(force_hists)):
    plt.plot(tvec,time_hists[i][:,2])

plt.figure()
for i in range(len(force_hists)):
    plt.plot(tvec,time_hists[i][:,4])

plt.figure()
for i in range(len(force_hists)):
    plt.plot(tvec,force_hists[i][:])

plt.show()