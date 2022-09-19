using JuMP
using GLPK
using Random
using Distributions

Random.seed!(42)

SCENARIOS = 100
COST_UNCERTAINTY = Normal(0, 64)
PACKAGE_LIMIT = [
    3 1 1
]
BASE_COST = [
    100 120 150
]

U = [
    150 115 50
    80 100 200
]

C = [
    c .+ rand(COST_UNCERTAINTY, SCENARIOS) for c in BASE_COST
]

I = 1:size(U)[1]
J = 1:length(C)
W = 1:SCENARIOS

model = Model(GLPK.Optimizer)

@variable(model, y[I, J, W], Bin)
@constraint(model, WillingUser[i in I, j in J, w in W], y[i, j, w] * (U[i, j] - C[j][w]) >= 0)
@constraint(model, LimitedNumberOfPackages[j in J, w in W], sum(y[:, j, w]) <= PACKAGE_LIMIT[j])
@objective(model, Max, sum(y[i, j, w] * (U[i, j] - C[j][w]) / SCENARIOS for i in I, j in J, w in W))
optimize!(model)

@show termination_status(model)
@show objective_value(model)
yOpt = value.(y)
packages_sold = reduce(+, yOpt[:,:,w] for w in W) ./ SCENARIOS

