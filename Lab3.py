# Lab3-probability.ipynb
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import time
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

def throw_a_coin(N):
    return np.random.choice(['Krona','Klave'], size=N)
N=40
throws = throw_a_coin(N)
print "Throws:"," ".join(throws)
print "Number of Krona:", np.sum(throws=="Krona")
print "p1 = Number of Krona/Total Throws:", np.sum(throws=='Krona')/float(N)

throws = throw_a_coin(N)
print "Throws:"," ".join(throws)
print "Number of Krona:", np.sum(throws=="Krona")
print "p2 = Number of Krona/Total Throws:", np.sum(throws=='Krona')/float(N)

N = 10000
throws = throw_a_coin(N)
print "First 100 throws:"," ".join(throws)[:1000]
print "Number of Krona:", np.sum(throws=="Krona")
print "p for 10,000 = Number of Krona/Total Throws:", np.sum(throws=='Krona')/float(N)

trials = [10, 20, 50, 70, 100, 200, 500, 800, 1000, 2000, 5000, 7000, 10000]
plt.plot(trials, [np.sum(throw_a_coin(j)=='Krona')/np.float(j) for j in trials], 'o-', alpha=0.6)
plt.xscale("log")
plt.axhline(0.5, 0, 1, color = "r")
plt.xlabel("number of trials")
plt.ylabel("probability of krona from simulation")
plt.title("frequentist probability of heads")
plt.show()

predictwise = pd.read_csv("predictwise.csv").set_index("States")
predictwise.head()

def simulate_election(model, n_sim):
    simulations = np.random.uniform(size=(51, n_sim))
    obama_votes = (simulations < model.Obama.values.reshape(-1, 1)) * model.Votes.values.reshape(-1, 1)
    return obama_votes.sum(axis=0)

result = simulate_election(predictwise, 10000)
print (result >= 269).sum()
result

def plot_simulation(simulation):
    plt.hist(simulation, bins=np.arange(200, 538, 1),
             label="simulations", align="left", normed=True)
    plt.axvline(332, 0, .5, color='r', label='Actual Outcome')
    plt.axvline(269, 0, .5, color='k', label='Victory Threshold')
    p05 = np.percentile(simulation, 5.)
    p95 = np.percentile(simulation, 95.)
    iq = int(p95-p05)
    pwin = ((simulation >= 269).mean() * 100)
    plt.title("Chance of Obama Victory: %0.2f%%, Spread: %d votes" % (pwin, iq))
    plt.legend(frameon=False, loc='upper left')
    plt.xlabel("Obama Electoral College Votes")
    plt.ylabel("Probability")
    sns.despine()

plot_simulation(result)




# Lab3-Freq.ipynb




# Lab3-Stats.ipynb