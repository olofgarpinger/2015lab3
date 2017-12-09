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

from scipy.stats import bernoulli
brv = bernoulli(p=0.3)
brv.rvs(size=20)

event_space=[0,1]
plt.figure(figsize=(12,8))
colors=sns.color_palette()
for i, p in enumerate([0.1, 0.2, 0.5, 0.7]):
    ax = plt.subplot(1, 4, i+1)
    plt.bar(event_space, bernoulli.pmf(event_space, p), label=p, color = colors[i], alpha = 0.5)
    plt.plot(event_space, bernoulli.cdf(event_space, p), color = colors[i], alpha=0.5)

    ax.xaxis.set_ticks(event_space)

    plt.ylim((0,1))
    plt.legend(loc=0)
    if i == 0:
        plt.ylabel("PDF at $k$")
plt.tight_layout()

CDF = lambda x: np.float(np.sum(result < x))/result.shape[0]
for votes in [200, 300, 320, 340, 360, 400, 500]:
    print "Obama Win CDF at votes=", votes, " is ", CDF(votes)

votelist = np.arange(0,540, 5)
plt.plot(votelist, [CDF(v) for v in votelist], '.-')
plt.xlim([200,400])
plt.ylim([-0.1,1.1])
plt.xlabel("votes for Obama")
plt.ylabel("probability of Obama win")

from scipy.stats import binom
plt.figure(figsize=(12,6))
k = np.arange(0, 200)
for p, color in zip([0.1, 0.3, 0.5, 0.7, 0.9], colors):
    rv = binom(200, p)
    plt.plot(k, rv.pmf(k), '.', lw=2, color=color, label=p)
    plt.fill_between(k, rv.pmf(k), color=color, alpha=0.5)
q=plt.legend()
plt.tight_layout()
q = plt.ylabel("PDF at $k$")
q = plt.xlabel("$k$")



# Lab3-Freq.ipynb
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

df = pd.read_table("babyboom.dat.txt", header=None, sep='\s+', names=['24hrtime','sex','weight','minutes'])
df.head()
df.minutes.mean()

df.corr()

g = sns.FacetGrid(col="sex", data=df, size=8)
g.map(plt.hist, "weight")

f = lambda x, l: l*np.exp(-l*x)*(x>0)
xpts=np.arange(-2,3,0.1)
plt.plot(xpts, f(xpts, 2),'o')
plt.xlabel("x")
plt.ylabel("exponential pdf")

from scipy.stats import expon

x = np.linspace(0, 4, 100)
colors = sns.color_palette()

lambda_ = [0.5, 1, 2, 4]
plt.figure(figsize=(12,4))
for l, c in zip(lambda_, colors):
    plt.plot(x, expon.pdf(x, scale=1./l), lw=2,
             color=c, label = "$\lambda = %.1f$"%l)
    plt.fill_between(x, expon.pdf(x, scale=1./l), color = c, alpha = .33)

plt.legend()
plt.ylabel("PDF at $x$")
plt.xlabel("$x$")
plt.title("Probability density function of an Exponential random variable;\
 differing $\lambda$")

from scipy.stats import expon
plt.plot(xpts, expon.pdf(xpts, scale=1./2.), 'o')
plt.hist(expon.rvs(size=1000, scale=1./2.), normed=True, alpha=0.5, bins=30)
plt.xlabel("x")
plt.title("exponential pdf and samples(normalize)")

rv = expon(scale=0.5)
plt.plot(xpts, rv.pdf(xpts),'o')
plt.hist(rv.rvs(size=1000), normed=True, alpha = 0.5, bins=30)
plt.plot(xpts, rv.cdf(xpts))
plt.xlabel("x")
plt.title("exponential pdf, cdf and samples(normalized)")

timediffs = df.minutes.diff()[1:]
timediffs.hist(bins=20)

lambda_from_mean = 1./timediffs.mean()
print lambda_from_mean, 1./lambda_from_mean

minutes=np.arange(0, 160, 5)
rv = expon(scale=1./lambda_from_mean)
plt.plot(minutes,rv.pdf(minutes),'o')
timediffs.hist(normed=True, alpha=0.5)
plt.xlabel("minutes")
plt.title("Normalized data and model for estimated $\hat{\lambda}$")

# Lab3-Stats.ipynb



















