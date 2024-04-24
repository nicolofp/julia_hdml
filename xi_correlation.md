ξ correlation coefficient
================
Nicoló Foppa Pedretti

### Introduction

ξ, a relatively novel correlation coefficient, surpasses classical
measures in detecting associations that lack monotonicity. Derived from
rank, ξ exhibits resilience against outliers, while its interpretation
as a measure of X and Y dependence remains straightforward. Its range
spans from 0, denoting independence, to 1, indicating dependence.
Moreover, it boasts a simple asymptotic theory applicable to sample
sizes as modest as 20 under the independence hypothesis. Even
categorical variables can undergo analysis through integer conversion. ξ
outperforms alternative tests in identifying oscillatory signals.
Despite these strengths, its only drawback arises in less power compared
to other independence tests for nonoscillatory signals in small samples.

$$y_i = \alpha + \beta x_i \qquad i = 1,\ldots,N$$

<details>
<summary>Code</summary>

``` julia
using Distributions, Plots, DataFrames, MarkdownTables

N = 500
x = sort(rand(Uniform(-5.0,5.0),N))
y = -0.4 .+ 2.926 .* x 
yhat = y + rand(Normal(0.0,1.0),N)
w = 4.0 .- 0.87 .* x.^2 
what = w + rand(Normal(0.0,1.0),N)

#=q1 = scatter(x,yhat, label = :none, title = "Regression line")
q1 = plot!(x,y, mc = :orange)
q2 = scatter(x,what, label = :none, title = "Quadratic line")
q2 = plot!(x,w, mc = :orange)
plot(q1, q2, layout=(1,2), size=(750,300))=#

X = DataFrames.DataFrame((; x,y,w,yhat,what))
#first(X,5) |> markdown_table()
```

</details>
### Resources

- https://arxiv.org/pdf/1909.10140
- https://souravchatterjee.su.domains/beam-correlation-trans.pdf
- https://www.linkedin.com/pulse/correlation-coefficient-xi-justin-bloesch-zxukc/
- https://github.com/jlbloesch/miscellaneous/blob/main/xicor.py
