# pymc3 Cheatsheet



## Model

### Model Setup & Prior Predictive Data

```python
X_obs = df[['x1', 'x2', 'x3']].values
y_obs = df['y'].values
# For categorical target
y_obs = pd.Categorical(df_obs['y']).codes

with pm.Model() as some_model:
    # priors
    beta_0 = pm.Normal('beta_0', mu=0, sigma=1.5)
    beta_distance = pm.Normal('beta_distance', mu=0, sigma=0.05)
    
    # Add data as pm.Data
    target = pm.Data('target', y_obs)
    
    # Interim calculations with pm.math
    some_value = pm.math.some_math_func(prior1, prior2)
    other_val = pm.Deterministic('some_value', some_value)
    
    # Likelihood
    targetl = pm.SomeDistribution('targetl',
                                  some_dist_parameter=other_val,
                                  observed=y_obs)
    # Sample prior predictive
    prior_pred = pm.sample_prior_predictive()
    
    # Get the data into an arviz xarray object
    idata = az.from_pymc3(prior=prior_pred)

# View the model structure
pm.model_to_graphviz(some_model)
```

### Sample Posterior Distributions

```python
n_tune = 2000
n_draws=1000

with some_model:
    trace = pm.sample(n_draws, tune=n_tune)
    # Add posterior sampling data to arviz object 
    idata.extend(az.from_pymc3(trace=trace))
```

### Sample Posterior Predictive

```python
with some_model:
    post_pred = pm.sample_posterior_predictive(trace)
    idata.extend(az.from_pymc3(posterior_predictive=post_pred))

# Save the inference data
idata.to_netcdf('dir/filename.nc')
```



## Basic Arviz Diagnostics

### Trace Plots

```python
trace_plots = az.plot_trace(idata.[prior, posterior, posterior_predictive],
                            var_names=['var1', 'var2', 'var3'],
                            combined=True,
                            compact=True,
                            coords={'chain': distance_azobj.prior.proba.chain.values,
                                    'draw': distance_azobj.prior.draw.values,
                                    'other_dim':  idata.prior.other_dim.values[0:100]},
                            figsize=(8,5))
trace_plots;
```

### Model Summary Stats

```python
summ = az.summary(idata.[prior, posterior, posterior_predictive])
summ

```

## Pandas

### Categorical dtype

```python
# Consider df[x1] series of length l with n categories
cat_data = pd.Categorical(df["x1"])
cat_data.codes
# returns an numpy int array length l where each category is coded to an integer between 0 and n
# [0, 0, 1, 1, 0, 3, 0, 1, ...]
cat_data.codes.shape
# (l, )
cat_data.categories
# returns an index array of length l with the value of each category
# Index(['cat1', 'cat2', 'cat3'])
cat_data.categories.nunique()
# returns l
```

