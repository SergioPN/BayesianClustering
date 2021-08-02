import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from dataLoad import dataLoad

data = dataLoad()
data.scaled_df

n_clusters = 3

X,y = data.scaled_df.body_mass_g.values.reshape(-1, 1), data.scaled_df.iloc[:, -1]


p = tfd.Dirichlet(tf.ones(n_clusters), name="probs_class")
categorical = tfd.Categorical(probs=p, name='categorical')


mu = tfd.HalfNormal(scale=[1]*n_clusters, name="mu")
sd = tfd.Normal(loc=[0]*n_clusters, scale=[1]*n_clusters, name="sd")

y = tfd.Normal(name="y", loc=mu, scale=sd)

