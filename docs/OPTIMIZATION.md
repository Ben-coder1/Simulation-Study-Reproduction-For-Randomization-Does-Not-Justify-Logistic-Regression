# Optimization 1


The majority of the time is spent fitting logistic regression

I applied an algorithmic improvement which reduces redundant computations (I only fit once rather than twice).

## Before

```python
estimator_results = {}

for estimator in self.param_estimators:
    estimator_name = type(estimator).__name__

    estimates, auxiliary = (
        estimator.generate_parameter_estimates(
            experiment_results
        )
    )

    estimator_results[estimator_name] = {
        "estimates": estimates,
        "auxiliary": auxiliary,
    }
```

## After

```python
plug_in_estimates, beta_mat = (
    self.param_estimator.generate_parameter_estimates(
        experiment_results
    )
)

logit_estimates = beta_mat[:, 1].copy()

# Preserve the original result names and structure.
estimator_results = {
    "LogitModelPlugIn": {
        "estimates": plug_in_estimates,
        "auxiliary": beta_mat,
    },
    "LogitModelEstimator": {
        "estimates": logit_estimates,
        "auxiliary": beta_mat,
    },
}
```
## Impact 
This improved runtime by about 1.9x for the original simulation suite. No impact on memory usage or stability gains

The main trade off is in code readability and extensibility. Before, my simulation method took in a list of methods, but now I instead just always use this one method, and do not take them in as separate methods.

This improvement can be seen os original vs non-vectorized. You can see how it improves for a given simulation setup.

![Performance comparisons of original, optimized/vectorized, and non-vectorized methods](../performance_results/runtime/timing_comparisons.png)

Alternatively, here you can see the overall run time comparisons.

![Overall run time comparisons](../performance_results/benchmarks/benchmark_comparison.png)

# Optimization 2

My other improvement was to vectorize the logistic regression fits over experiments. This was to solve the same problem as before. Rather than fitting a new logit model for each experiement, I did it all at once using vectorization.

## Before

```python
for exp_result in experiment_results:
    ...
    fitted_result = sm.Logit(outcomes, X).fit(disp=False)
    plug_in_estimate = self.plug_in_differential_log_odds(
        fitted_result,
        covariates,
    )

    beta_mat.append(fitted_result.params)
    plug_in_estimates.append(plug_in_estimate)
```

## After

```python
for experiment_index, experiment_result in enumerate(experiment_results):
    ...
    assignment_matrix[experiment_index] = assignments
    outcome_matrix[experiment_index] = outcomes

...

for _ in range(self.max_iter):
    ...
    newton_steps = np.linalg.solve(
        hessian_matrices,
        score_vectors[..., None],
    )[..., 0]
    ...
```

## Impact

As you can see from the charts it runs faster, however it takes up a lot more memory. This can be seen by the associated charts.



![Memory comparisons of original, optimized/vectorized, and non-vectorized methods](../performance_results/memory/memory_comparisons.png)

However, the speedup depends drastically on what computer you are using. The charts you see above were all from my mac, while my older pc actually had much less speedup, including no improvement when simulating 5000 individuals.

One other thing to note is that this code, when run on my mac (but not my windows), often produces the following: "Warnings: RuntimeWarning: divide by zero encountered in matmul | RuntimeWarning: invalid value encountered in matmul | RuntimeWarning: overflow encountered in matmul". This is a known mac bug for the particular numpy edition, and is not evidence of convergence or numerical issues, especially as my regression tests pass, and doing matrix multiplication of identity matrices gives those identical warnings as well. 



# Optimization 3


This handles the same issue/bottleneck as before.



I applied parallelization to the logistic regression fitting.

## Code Change


```python
 def generate_parameter_estimates(
        self,
        experiment_results: list[ExperimentResults],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_chunks = min(self.workers, len(experiment_results))
        estimate = self.estimator.generate_parameter_estimates
        if n_chunks <= 1:
            return estimate(experiment_results)

        chunk_size, remainder = divmod(len(experiment_results), n_chunks)
        chunks = []
        start = 0
        for chunk_index in range(n_chunks):
            stop = start + chunk_size + (chunk_index < remainder)
            chunks.append(experiment_results[start:stop])
            start = stop

        # map yields results in chunk order even if a later chunk finishes first.
        plug_in_estimates, coefficients = zip(*self.executor.map(estimate, chunks))
        return (
            np.concatenate(plug_in_estimates),
            np.concatenate(coefficients, axis=0),
        )
```

## Impact

It improved runtime over the vectorized but non-parallel by about 1.4x. Refer to the previous charts for comparisons. It has the same memory usage, and has greater timing improvemenets when there are more experiments, and more individuals. It is actually slower when there are few individuals and few experiements. 



# LESSONS LEARNED
Removing duplicated code was the best return on investment since it gave a sizeable performance boost for minimal results. I was minimally surprised about where time was spent. I rogiinally planned to vectorize the experiment creations, but since the profiling showed that very little time was spent there I decided against it. In general, I think any optimziatinos regarding the DGP would not be worth it here.

The other improvmeent that was a great return on investmenet was parallelization, as it provided noticeable improvement without too much extra work. 

One other thing I learned is that an optimization may have different levels of effect on different hardware. While obviously possible, I did not expect such thigns to happen here. 
