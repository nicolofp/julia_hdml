using Dates, DataFrames, Plots, StatsPlots, Statistics, Distributions
using StatsBase, HypothesisTests, LinearAlgebra, Random, MLJ, CSV, CategoricalArrays 

DT = CSV.read("C:/Users/nicol/Documents/solar_1/Solar Power Plant Data.csv", DataFrame);

show(describe(DT),allcols = true)

DT[DT.Radiation .< 0,:Radiation] .= 0.0;

DT.day = parse.(Int64,chop.(DT[:,"Date-Hour(NMT)"], head = 0, tail = 14))
DT.month = parse.(Int64,chop.(DT[:,"Date-Hour(NMT)"], head = 3, tail = 11))
DT.hour = parse.(Int64,chop.(DT[:,"Date-Hour(NMT)"], head = 11, tail = 3));

rename!(DT,"Date-Hour(NMT)" => "timedate");

DT.timedate_real = DateTime.(2017,DT.month,DT.day,DT.hour);
DT.date_real = Date.(2017,DT.month,DT.day);

DT[:,"h_light"] .= zeros(size(DT,1))
DT[DT.Radiation .!= 0,:h_light] .= 1.0;

# hours of light
# 

df = groupby(DT, :date_real)
dt = combine(df, 
             ["SystemProduction","WindSpeed","Sunshine","AirPressure",
              "Radiation","AirTemperature","RelativeAirHumidity","h_light"] .=> [sum, mean, mean, mean, mean, 
                                                                                 mean, mean, sum]; 
    renamecols = true);
sort!(dt,:date_real);

p1 = scatter(DT.Radiation, DT.SystemProduction, title = "Solar Radiation (day)", label = :none)
p2 = scatter(DT.Sunshine, DT.SystemProduction, title="Sunshine (day)", label = :none, mc=:orange)
p3 = scatter(DT.Radiation,DT.Sunshine, title = "Sun correlation (day)", label = :none)
p4 = plot(DT.timedate, DT.SystemProduction, title = "System production (daily)", label = :none)
plot(p1, p2, p3, p4, layout=(2,2), legend=false, size=(900,600))

dt.is_wierd = zeros(size(dt,1))
dt[(dt.Radiation_mean .> 33) .& (dt.SystemProduction_sum .< 5000),:is_wierd] .= 1.0;
dt[dt.SystemProduction_sum .== 0,:is_wierd] .= 1.0;

DTF = leftjoin(DT,dt, on = :date_real);
DTF.hour = coerce(DTF.hour, OrderedFactor);

df = DTF[(DTF.is_wierd .== 0),:] # .& (DTF.hour .== 12),:]
p1 = scatter(df.Radiation, df.SystemProduction, title = "Solar Radiation (day)", label = :none)
p2 = scatter(df.Sunshine, df.SystemProduction, title="Sunshine (day)", label = :none, mc=:orange)
p3 = scatter(df.RelativeAirHumidity, df.SystemProduction, title="RH (day)", label = :none, mc=:green)
p4 = scatter(df.hour, df.SystemProduction, title="LH vs Production (day)", label = :none, mc=:red)
plot(p1, p2, p3, p4, layout=(2,2), legend=false, size=(800,600))

LGBMRegressor = @load LGBMRegressor
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
rf_model = RandomForestRegressor();

dt_ml = DTF[(DTF.is_wierd .== 0) .& (DTF.Radiation .> 0),:];

train, test = (collect(1:2500),collect(2501:3585));

X = MLJ.table(Matrix{Float64}(dt_ml[:,vcat(2:7,11)]))
y = dt_ml[:,8];

rf1 = range(rf_model, :min_samples_leaf, values = [5, 20, 50, 100])
rf2 = range(rf_model, :n_subfeatures, lower = 1, upper = 7)
rf3 = range(rf_model, :n_trees, values = [50, 100, 250, 500])
rf_tm = TunedModel(model = rf_model, 
                   tuning = Grid(resolution = 5), # RandomSearch()
                   resampling = CV(nfolds = 3, rng = 123), 
                   ranges = [rf1,rf2,rf3],
                   measure = rms)
rf_mtm = machine(rf_tm, X, y)
fit!(rf_mtm, rows = train);

rf_best_model = fitted_params(rf_mtm).best_model
@show rf_best_model.min_samples_leaf
@show rf_best_model.n_subfeatures
@show rf_best_model.n_trees;

evaluate!(rf_mtm, resampling = CV(nfolds=10, rng=1234), measure = [rms, rsquared], rows = train)

predictions = MLJ.predict(rf_mtm, rows=test);

Mprod = maximum(dt_ml[test,:SystemProduction])
q1 = scatter(dt_ml[test,:SystemProduction],predictions, label = :none, title = "Actual vs Predict")
q1 = plot!(collect(0:Mprod),collect(0:Mprod), label = :none)
q2 = plot(dt_ml[test,:date_real],dt_ml[test,:SystemProduction],  title = "Actual vs Predict", label = ["Actual"])
q2 = plot!(dt_ml[test,:date_real],predictions, mc = :orange, label = ["Predict"])
plot(q1, q2, layout=(1,2), legend=false, size=(900,300))

mape = 100 * mean(abs.((dt_ml[test,:SystemProduction] - predictions) ./ dt_ml[test,:SystemProduction])) 

println("RMSE: ", string.(rmse(dt_ml[test,:SystemProduction],predictions)))
println("MAE: ", string.(mae(dt_ml[test,:SystemProduction],predictions)))
println("R²: ", string.(cor(dt_ml[test,:SystemProduction],predictions).^2))
println("MAPE: ", string.(mape))

forecast = dt_ml[test,[:date_real,:timedate,:hour,:SystemProduction]]
forecast.pred = predictions

fc = groupby(forecast, :date_real)
forecast_day = combine(fc, 
             ["SystemProduction","pred"] .=> [sum, sum]; 
    renamecols = true);
sort!(dt,:date_real);

plot(forecast_day[:,:date_real],forecast_day[:,:SystemProduction_sum],  title = "Actual vs Predict", label = ["Actual"], 
     legend=false, size=(450,300))
plot!(forecast_day[:,:date_real],forecast_day[:,:pred_sum], mc = :orange, label = ["Predict"])

mape = 100 * mean(abs.((forecast_day[:,:SystemProduction_sum] - forecast_day[:,:pred_sum]) ./ forecast_day[:,:SystemProduction_sum])) 

println("RMSE: ", string.(rmse(forecast_day[:,:SystemProduction_sum],forecast_day[:,:pred_sum])))
println("MAE: ", string.(mae(forecast_day[:,:SystemProduction_sum],forecast_day[:,:pred_sum])))
println("R²: ", string.(cor(forecast_day[:,:SystemProduction_sum],forecast_day[:,:pred_sum]).^2))
println("MAPE: ", string.(mape))
