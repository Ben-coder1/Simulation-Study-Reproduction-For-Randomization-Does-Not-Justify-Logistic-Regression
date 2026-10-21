## ADEMP Simulation Description

|  |  |
|---|---|
| **Aims** | See the performance of the logit model and plug-in estimator under experimental data when the data are not generated from a logit model and the analysis is done under Neyman’s nonparametric setup. This includes looking specifically at standard deviation and bias. |
| **DGM: Model structure** | For $i\in[n]$, generate i.i.d. $\operatorname{Uniform}(0,1)$ random variables $U_i$ and $V_i$, and set $Y_i^C=\mathbf{1}(U_i>1/2)$ and $Y_i^T=\mathbf{1}(U_i+V_i>3/4)$. Then, with those values frozen, conduct $n_{\mathrm{sim}}=1000$ randomized experiments. In each experiment, randomly assign exactly $75\%$ of $[n]$ to the treatment group and $25\%$ to the control group. The observed outcome for individual $i$ is $Y_i^T$ if assigned to treatment and $Y_i^C$ if assigned to control. The data reported are $V_i$, the assignments, and the observed outcomes. The number of individuals are $n\in\{100,500,1000,5000\}$. |
| **Estimand** | Treatment-minus-control differential log odds of success: $\Delta_n=\operatorname{logit}(\overline{Y^T})-\operatorname{logit}(\overline{Y^C})$. |
| **Methods** | Treatment-assignment coefficient of the logit model fitted by maximum likelihood estimation (MLE); plug-in estimator presented in *Randomization Does Not Justify Logistic Regression* by David A. Freedman which uses predictions from the same fitted logistic-regression model. |
| **Performance measures** | Standard deviation and bias. For estimates $\widehat{\theta}_1,\ldots,\widehat{\theta}_{n_{\mathrm{sim}}}$, let $\overline{\widehat{\theta}}=n_{\mathrm{sim}}^{-1}\sum_{r=1}^{n_{\mathrm{sim}}}\widehat{\theta}_r$. The standard deviation is estimated using $\widehat{\operatorname{SD}}(\widehat{\theta})=\sqrt{\frac{1}{n_{\mathrm{sim}}-1}\sum_{r=1}^{n_{\mathrm{sim}}}\left(\widehat{\theta}_r-\overline{\widehat{\theta}}\right)^2}$. The sample bias is the difference between the average estimate and the true differential log odds: $\widehat{\operatorname{Bias}}(\widehat{\theta})=\overline{\widehat{\theta}}-\Delta_n$. |

## Design Table

| Simulation | Number of Individuals |
|---|---:|
| Simulation 1 | $n=100$ |
| Simulation 2 | $n=500$ |
| Simulation 3 | $n=1000$ |
| Simulation 4 | $n=5000$ |