# Package Price Prediction and Auction Optimization

Online marketplaces have grown a lot during the last years. They essencially match users to produts.However, considering that both users financial power and the amount of products are limited, how should a system recommend products and calculate how much to charge?

This repository is a quick study on how to optimized the match between uncertain predicted costs (how much?) and product availability (how many can be sold?) to users (who will buy? - and how much should be charged?).

## Problem

A simple example that can be expanded on is matching clients to ice cream flavors. Each user has a flavor preference, and the cost of ice creams depends on its package and can change based on flavor - e.g. pistacchio ice cream can have a much greater cost than a vanilla one.

Moreover, the future prices of resources are not certain: if cocoa plantations get sick, chocolat ice cream can get much more expansive to produce.

Therefore, predicting what could happen - regarding production, transportation, etc. - and how that will affect costs is very important to keep client satisfaction and margins.

## Roadmap

- [ ] Data
  - [ ] Generate time series data with uncertainty (use [timeseries-generator](https://github.com/Nike-Inc/timeseries-generator))
  - [ ] Use real world data for the problem
- [ ] Model
  - [ ] Use given data to generate predictions intervals
  - [ ] Test
- [ ] Optimization
  - [x] Test optimization model
  - [ ] Optimize a real problem by sampling model predictions from its interval of a real problem
  - [ ] Consider discounts for users: $y_i^j$ does not need to be binary ([define specific discrete set for JuMP variable](https://discourse.julialang.org/t/how-to-define-a-jump-variable-in-a-specific-discrete-set/42754))
  - [ ] Model user preference: transform $U_i^j$ into a function of user *i* and product *j*
- [ ] Other
  - [ ] Add projects structure
  - [ ] Improve problem section with a more realistic challenge

## Modelling

Considering the [problem](#problem) presented, there are three main variables that must be modelled:

- How many of each product is available;
- How much each product costs to produce; and
- How much are users willing to pay for each product

It is clear that each one of those have uncertanties related to them and can be very difficult to estimate e.g. how much users are willing to pay. Furthermore, the cost of a product can have lots of factors - possibily with correlations between them -, making it harder to predict.

With this in mind, some challenges are:

- Calculating, understading and communicating the Prediction Interval; and
- Calculating the Prediction Interval for a sum of factors ([discussion in cross-validated](https://stats.stackexchange.com/questions/14987/the-sum-of-prediction-intervals))

## Optimization

To match which packages are sold to whom based on how many there are available, an optimization model is used, present below. It maximizes profits restricted to the fact that packages are only sold if an user is willing to pay for it more than it would cost.

It does not consider some important steps:

- Users pricing: the maximum amount that an user would pay is known and is always the package's price;
- Financial limit: users do not have a financial limit, enabling them to buy as many packages as they would; and
- Recommendation: if an user would buy the package, it will be recommended and it will be bought;

$$
\begin{aligned}
    \text{MAX}_{y_i^j} \quad & \Sigma_{i \in I,  j \in J}  y_i^j \cdot (U_i - C_j) \\
    y_i^j \cdot (U_i^j - C_j) &\ge 0 \quad \forall i \in I,  j \in J \\
    \Sigma_{i \in I} y_i &\le \text{limit}_j \quad \forall j \in J \\
    y_i^j &\in \{0, 1\}
\end{aligned}
$$

where:

- $C_j$ is the cost of package *j*;
- $\text{limit}_j$ restricts the amount of packages *j* that are available;
- $U_i^j$ is how much an user is willing to pay for package *j*; and
- $y_i^j$ is an indicator that user *i* bought package *j*

Although this models considers that $U_i^j$ and $C_j$ to be deterministic (have a fixed value), it can be expanded to consider different scenarios - e.g. have uncertainty related to its values. Solving these cenarios and averaging them out is a technique called [Sample Average Approximation](https://link.springer.com/article/10.1023/A:1021814225969), which can be used to approximate any real problem given enough scenarios. An example that expands uncertainty for $C_j$ can be found in [main.jl](./src/main.jl), where this model is implemented.

### Results example

Consider that there are `2 users` and `3 packages`. The variables discussed under [modelling](#modelling) are here represented by:

- `PACKAGE_LIMIT` limits how many packages of each type can be sold (there is only one "package 2" available);
- `C` represents the base cost (prediction expected value) of each package ("package 3" expected cost is $150); and
- `U` represents the maximum chargeable amount that each user would pay for each package ("user 2" would pay a maximum of $100 for "package 2" - thus it would not buy it as $150 > $100)

```julia
# set limit_i
PACKAGE_LIMIT = [
    3 1 1
]
# set C_j
BASE_COST = [
    100 120 150
]
# set U_i^j
U = [
    150 115 50
    80 100 200
]
```

This example illustrates `SCENARIOS` scenarios of uncertain cost for each product, assuming that its uncertainty distribution is normally distributed.

```julia
# Sample Average Approximation
SCENARIOS = 100 # amount of scenarios
COST_UNCERTAINTY = Normal(0, 64) # uncertainty distribution

# Uncertain costs
C = [
    c .+ rand(COST_UNCERTAINTY, SCENARIOS) 
    for c in BASE_COST
]

# ... optimization model
```

Finally the results indicate that:

- The expected profit for the expected scenarios is 161.99;
- "package 1" costs less then or equal the amount that "user 1" is willing to pay for it 86% of the times, which follows for other entries.

```julia
# results
# profit = 161.99
#     1     2     3
# 1   0.86  0.57  0.0
# 2   0.41  0.0   0.72
```

Something important to notice is that neither "user 1" buys "package 3" nor "user 2" buys "package 2". This happens because there are only one available for each, and each user pays more than the other for the other product, making it unavaible for the other to buy.
