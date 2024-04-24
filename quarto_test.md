Quarto test usign Julia
================
Nicoló Foppa Pedretti

## Example 1

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
first(X,5) |> markdown_table()
```

</details>

| x                  | y                   | w                   | yhat                | what                |
|--------------------|---------------------|---------------------|---------------------|---------------------|
| -4.967102067417719 | -14.933740649264248 | -17.46472956488648  | -16.243470725611733 | -16.395645256183123 |
| -4.934475710279944 | -14.838275928279117 | -17.1836739657482   | -14.77994291865972  | -19.16599249606457  |
| -4.889138786899427 | -14.705620090467725 | -16.79619992748103  | -14.978741561447906 | -16.750359171347604 |
| -4.836700229241487 | -14.552184870760591 | -16.352492123563845 | -12.873285929919666 | -17.22252557536072  |
| -4.825979871766952 | -14.520817104790103 | -16.262371098748797 | -13.317210412748988 | -16.754797628262995 |
