using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random

# Dataset Test
N = 500
rng1 = MersenneTwister(2090);
rng2 = MersenneTwister(1990);
rng3 = MersenneTwister(3022);
x = rand(rng1,Uniform(-4.0,4.0),N,5)
a1 = -1.2
a2 = 0.95
b1 = vec([1.1 -0.12 0.34 2.13 -1.8])
b2 = vec([2.1 -1.12 0.94 -1.13 -0.38])
y1 = a1 .+ x * b1 + rand(rng2,Normal(0.0,0.5),N)
y2 = a2 .+ x * b2 + rand(rng3,Normal(0.0,0.5),N)

dt = DataFrame(hcat(y1,y2,x), ["y1","y2","x1","x2","x3","x4","x5"])

out = ["y1","y2"] 
cov = ["x1","x2","x3","x4","x5"]

function lm_ewas(dt, out, cov)
    X_tmp = hcat(ones(size(dt,1)), Matrix{Float64}(dt[:,cov]))
    Y_tmp = Matrix{Float64}(dt[:,out])
    β = X_tmp\Y_tmp 
    σ = sqrt.(vec(sum((Y_tmp - X_tmp*β).^2,dims = 1)./(size(X_tmp,1)-size(X_tmp,2))))
    Σ = inv(X_tmp'*X_tmp)
    std_coeff = sqrt.(kron(σ,diag(Σ))) # Kronecker product

    βvec = reshape(β,length(out) * (length(cov) + 1))
    tval = βvec ./ std_coeff
    pval = cdf(TDist(size(X_tmp,1)-size(X_tmp,2)), -abs.(βvec ./ std_coeff))
    ci025 = βvec .- quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975) .* std_coeff
    ci975 = βvec .+ quantile(TDist(size(X_tmp,1)-size(X_tmp,2)), 0.975) .* std_coeff    
    outcome = repeat(out, inner = length(cov) + 1)
    covariates = repeat(vcat("intercept",cov), outer = length(out))

    tmp = (outcome = outcome, covariates = covariates,
           beta = βvec, sd = std_coeff, tval = tval, 
           pval = pval, ci025 = ci025, ci975 = ci975)
    return tmp
end 

@time results = lm_ewas(dt, ["y1","y2"], ["x1","x2","x3","x4"]);

