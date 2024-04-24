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

| x                   | y                   | w                   | yhat                | what                |
|---------------------|---------------------|---------------------|---------------------|---------------------|
| -4.998955617929383  | -15.026944138061376 | -17.740914824924133 | -14.210439033859377 | -17.182324593690694 |
| -4.9804139799385805 | -14.972691305300287 | -17.579935368063857 | -15.625484672960939 | -17.548255763172623 |
| -4.968656993907067  | -14.93829036417208  | -17.478170521098402 | -14.349790899258325 | -16.973753080844887 |
| -4.9607022639174145 | -14.915014824222355 | -17.409453247574763 | -13.632811911240447 | -16.87581308732164  |
| -4.953309063951904  | -14.893382321123273 | -17.345685494234438 | -14.126946284912147 | -17.88944873013607  |
