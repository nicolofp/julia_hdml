Quarto test usign Julia
================
Nicoló Foppa Pedretti

## Example 1

Test function

$$y_i = \alpha + \beta x_i \qquad i = 1,\ldots,N$$

$$ \xi_n(X,Y) = 1 - y_w $$

$$\frac{3 \sum_{i=1}^{n-1} |r_{i+1}-r_i| }{n^2 - 1} $$

$$3 \sum_{i=1}^{n-1} |r_{i+1}-r_i| $$

$$\sum_{n=1}^{\infty} 2^{-n} = 1$$

## Code

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
