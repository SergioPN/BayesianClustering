%load_ext autoreload
%autoreload 2
from dataLoad import dataLoad
import pymc3 as pm
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns

data = dataLoad()

n_clusters = 3
n_observations, n_features = data.scaled_df.shape

with pm.Model() as model:
    p = pm.Dirichlet("p", a=np.ones(n_clusters))
    category = pm.Categorical("category", p=p, shape=n_observations)

    bm_sigmas = pm.HalfNormal("bm_sigmas", sigma=1, shape=n_clusters)
    bm_means = pm.Normal("bm_means", np.zeros(n_clusters), sd=1, shape=n_clusters)

    y_bm = pm.Normal("y_bm", mu=bm_means[category], sd=bm_sigmas[category], observed=data.scaled_df.body_mass_g)
    trace_base = pm.sample(10000)

groups = trace_base.get_values("category", burn=6000, combine=True)[200]

data.scaled_df["groups"] = pd.Categorical(groups)


alt.Chart(data.df).mark_bar().encode(
    x=alt.X('body_mass_g:Q',bin=alt.Bin(maxbins=10)),
    y = 'count()',
    color = 'species:N'
)

# More features

# ~ 8 mins de entrenamiento
with pm.Model() as model_normal:
    p = pm.Dirichlet("p", a=np.ones(n_clusters))
    category = pm.Categorical("category", p=p, shape=n_observations)

    sigmas, means, ys = {}, {}, {}
    for col in data.scaled_df.columns[:4].values:
        sigmas[col] = pm.HalfNormal(f"{col}_sigmas", sigma=1, shape=n_clusters)
        means[col] = pm.Normal(f"{col}_means", np.zeros(n_clusters), sd=1, shape=n_clusters)
        ys[col] = pm.Normal(f"y_{col}", mu=means[col][category], sd=sigmas[col][category], observed=data.scaled_df[col])

    trace_medium = pm.sample(10000)

groups_med = trace_medium.get_values("category", burn=10, combine=True)[50]

data.scaled_df["groups_med"] = pd.Categorical(groups_med)


alt.Chart(data.scaled_df).mark_bar().encode(
    x=alt.X('groups_med:N'),
    y='count()',
    color='species:N'
)

# Advanced

n_features = 4

# ~ 60 mins .... :(
with pm.Model() as model_advanced:
    chol, corr, stds = pm.LKJCholeskyCov("chol", n=n_features, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True)
    cov = pm.Deterministic("cov", chol.dot(chol.T))

    mu = pm.Normal("mu", mu=0, sd=1.5, shape=(n_clusters, n_features), testval=data.scaled_df.iloc[:, :4].values.mean(axis=0))

    p = pm.Dirichlet("p", a=np.ones(n_clusters))
    category = pm.Categorical("category", p=p, shape=n_observations)

    y = pm.MvNormal("y", mu[category], chol=chol, observed=data.scaled_df.iloc[:, :4].values)

    trace_advanced = pm.sample(10000, cores=4)

groups_adv = trace_advanced.get_values("category", burn=6000, combine=True)
from scipy.stats import mode

groups_mode = mode(groups_adv, axis=0)[0].reshape(-1)
data.scaled_df["groups_mode"] = pd.Categorical(groups_mode)


alt.Chart(data.scaled_df).mark_bar().encode(
    x=alt.X('groups_mode:N'),
    y='count()',
    color='species:N'
)

from sklearn.metrics import f1_score
f1_score(data.scaled_df["species"].cat.codes, groups_mode, average=None)