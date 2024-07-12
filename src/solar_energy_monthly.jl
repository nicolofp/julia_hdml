# Load libraries and import  dataset
using Dates, DataFrames, Plots, StatsPlots, Statistics, Distributions
using StatsBase, HypothesisTests, LinearAlgebra, Random, MLJ, CSV, CategoricalArrays 

DT = CSV.read("C:/Users/nicol/Documents/solar_1/Solar Power Plant Data.csv", DataFrame);

show(describe(DT),allcols = true)

# Set all the negative radiation to zero
DT[DT.Radiation .< 0,:Radiation] .= 0.0;

# Parse properly all dates and hours
DT.day = parse.(Int64,chop.(DT[:,"Date-Hour(NMT)"], head = 0, tail = 14))
DT.month = parse.(Int64,chop.(DT[:,"Date-Hour(NMT)"], head = 3, tail = 11))
DT.hour = parse.(Int64,chop.(DT[:,"Date-Hour(NMT)"], head = 11, tail = 3));

# Rename column for convenience
rename!(DT,"Date-Hour(NMT)" => "timedate");

# Create column with right formatting
DT.timedate_real = DateTime.(2017,DT.month,DT.day,DT.hour);
DT.date_real = Date.(2017,DT.month,DT.day);

# Create variable "Hours of light"
DT[:,"h_light"] .= zeros(size(DT,1))
DT[DT.Radiation .!= 0,:h_light] .= 1.0;

# Analyze why I have zero Radiation but positive energy produced (--> to be investigated)
# Sunshine = "minutes per hours that sun is not cover by clouds" (scaled 0-60)
# Consider delay between radiation hit solar panel and start producing energy
tmp = DT[(DT.Radiation .== 0) .&& (DT.SystemProduction .> 0),:]
tmp2 = DT[(DT.Sunshine .== 0) .&& (DT.SystemProduction .> 0),:]


# hours of light
scatter(DT.Sunshine[1:8759], DT.Radiation[2:8760])
scatter(DT.Radiation[DT.month .== 3,:], DT.Sunshine[DT.month .== 3,:])
scatter(DT.Radiation[1:8759], DT.SystemProduction[2:8760])
scatter(DT.Radiation, DT.SystemProduction)
cor(Matrix(DT[DT.SystemProduction .!= 0,2:8]))
DT[DT.Radiation .<= 10,:]

mm = 5
mmax = 31
Pl = plot(DT[DT.date_real .== Date.(2017,mm,1),:hour], 
          DT[DT.date_real .== Date.(2017,mm,1),:SystemProduction],
          label = :none, lc = :orange)
for i in 2:mmax
    Pl = plot!(DT[DT.date_real .== Date.(2017,mm,i),:hour], 
          DT[DT.date_real .== Date.(2017,mm,i),:SystemProduction],
          label = :none, lc = :orange)
end
Pl




df = groupby(DT, :date_real)
dt = combine(df, 
             ["SystemProduction","WindSpeed",
             "Sunshine","AirPressure",
              "Radiation","AirTemperature",
              "RelativeAirHumidity","month"] .=> [sum, mean, mean, 
                                                            mean, mean, mean, 
                                                            mean,  mean]; 
    renamecols = true);
sort!(dt,:date_real);

# Analysis to detect anomalies in 
# daily production vs daily radiation 
# --> delete day with Radiation mean > 40 and production sum < 1800
scatter(dt.Radiation_mean, dt.SystemProduction_sum)
vline!([40])
hline!([1800])
suspect_day = dt[(dt.Radiation_mean .> 40) .&& (dt.SystemProduction_sum .< 1800),:date_real]
dt.is_suspect = zeros(size(dt,1))
dt[(dt.Radiation_mean .> 40) .&& (dt.SystemProduction_sum .< 1800),:is_suspect] .= 1.0;
zero_prod = dt[dt.SystemProduction_sum .== 0,:]

DF = DT[DT.date_real .∉ Ref(suspect_day),:]


scatter(DF[(DF.SystemProduction .== 0) .&& (DF.Radiation .> 10),:Radiation],
        DF[(DF.SystemProduction .== 0) .&& (DF.Radiation .> 10),:SystemProduction])
        
scatter(DF[(DF.Radiation .< 40),:Radiation],
        DF[(DF.Radiation .< 0),:SystemProduction])

filter!([:date_real, :SystemProduction, :Radiation] => (x,y,z) -> x ∉ Ref(suspect_day) && 
        !(y == 0 && z > 40), DT)

cor(dt[(dt.Radiation_mean .< 40) .|| (dt.SystemProduction_sum .> 1800),:Radiation_mean],
    dt[(dt.Radiation_mean .< 40) .|| (dt.SystemProduction_sum .> 1800),:SystemProduction_sum])
scatter(dt[(dt.Radiation_mean .< 40) .|| (dt.SystemProduction_sum .> 1800),:Radiation_mean],
        dt[(dt.Radiation_mean .< 40) .|| (dt.SystemProduction_sum .> 1800),:SystemProduction_sum])

scatter(DT[DT.date_real .∉ Ref(suspect_day),:Radiation][1:8135], 
        DT[DT.date_real .∉ Ref(suspect_day),:SystemProduction][2:8136])
cor(DT[DT.date_real .∉ Ref(suspect_day),:Radiation][1:8135], 
    DT[DT.date_real .∉ Ref(suspect_day),:SystemProduction][2:8136])

countmap(DT[DT.date_real .∉ Ref(suspect_day),:month])
histogram(DT[DT.SystemProduction .> 10,:SystemProduction])

function cyclical_encoder(df::DataFrame, columns::Union{Array, Symbol}, max_val::Union{Array, Int} )
    for (column, max) in zip(columns, max_val)        
        df[:, Symbol(string(column) * "_sin")] = sin.(2*pi*df[:, column]/max)
        df[:, Symbol(string(column) * "_cos")] = cos.(2*pi*df[:, column]/max)
    end
    return df
end

cyclical_encoder(DT, ["day","month","hour"], [31,12,23])


k = 0
# train, test = partition(collect(eachindex(DT.SystemProduction)), 0.75, shuffle=true, rng=111);
DT_model = DT[DT.Radiation .> k,:]
#train, test = partition(collect(eachindex(DT_model[:,:SystemProduction])), 
#                        0.80, shuffle=true, rng=90);
#train, test = partition(1:size(DT_model,1), 
#                        0.25, rng=90, shuffle = false);
# train, test = (collect(1:1564),collect(1565:3267));
train, test = (collect(1:1828),collect(1829:3744));
#X = MLJ.table(Matrix{Float64}(DT[:,2:7]));
X = DT_model[:,vcat(2:7,14:19)];
y = DT_model[:,:SystemProduction];

# First let's try to LM and the crossvalidation
LinearRegressor = @load LinearRegressor pkg=GLM
EvoTreeRegressor = MLJ.@load EvoTreeRegressor pkg=EvoTrees
LGBMRegressor = MLJ.@load LGBMRegressor pkg=LightGBM
RandomForestRegressor = MLJ.@load RandomForestRegressor pkg=DecisionTree
pipe_transformer = (X -> coerce!(X, :month=>OrderedFactor, 
                                    :hour=>OrderedFactor)) |> ContinuousEncoder()

et_regressor = EvoTreeRegressor(nbins = 32, max_depth = 10, nrounds = 200)
rf_regressor = RandomForestRegressor(n_trees = 500, n_subfeatures = 3, min_samples_leaf = 10)
lg_regressor = LGBMRegressor(learning_rate = 0.1, min_data_in_leaf = 10, num_iterations = 150)
#=et_bins = range(et_regressor, :nbins, values = [16, 32, 64])
et_nrun = range(et_regressor, :nrounds, values = [100, 250, 500])
et_deph = range(et_regressor, :max_depth, values = [4, 5, 6])
et_tm = TunedModel(model = et_regressor, 
                   tuning = Grid(resolution = 10), # RandomSearch()
                   resampling = CV(nfolds = 5, rng = 123), 
                   ranges = [et_bins, et_nrun, et_deph],
                   measure = [rmse])=#

model_glm = et_regressor
model_glm = lg_regressor
model_glm = rf_regressor
mach_glm = machine(model_glm, X, y) 
fit!(mach_glm, rows = train)

# Cross-validation
evaluate!(mach_glm, resampling = CV(nfolds=5, rng=1234), 
          repeats=5, measure = [rmse, rsquared])

# fitted_params(mach_glm).linear_regressor.coef
# report(mach_glm)
# residuals = y[train,:] - (520.21 .+ Matrix(X[train,:]) * fitted_params(mach_glm).linear_regressor.coef)
# histogram(residuals)
# scatter(residuals)

# scatter(MLJ.predict(mach_glm, rows=test),
#         y[test])

DT_model.predicts = zeros(size(DT_model,1))        
DT_model[test,:predicts] .= MLJ.predict(mach_glm, rows=test)
histogram(MLJ.predict(mach_glm, rows=test))

df = groupby(DT_model, :date_real)
dt = combine(df, ["SystemProduction","predicts"] .=> [sum, sum]; renamecols = true);
sort!(dt,:date_real);
dt = dt[dt.date_real .> Date.(2017,6,30),:]

q2 = plot(dt[:,:date_real],dt[:,:SystemProduction_sum],  title = "Actual vs Predict", label = ["Actual"])
q2 = plot!(dt[:,:date_real],dt[:,:predicts_sum], mc = :orange, label = ["Predict"])

mape = 100 * mean(abs.((dt[:,:SystemProduction_sum] - dt[:,:predicts_sum]) ./ dt[:,:SystemProduction_sum])) 

println("RMSE: ", string.(rmse(dt[:,:SystemProduction_sum],dt[:,:predicts_sum])))
println("MAE: ", string.(mae(dt[:,:SystemProduction_sum],dt[:,:predicts_sum])))
println("R²: ", string.(cor(dt[:,:SystemProduction_sum],dt[:,:predicts_sum]).^2))
println("MAPE: ", string.(mape))

histogram(dt[:,:SystemProduction_sum], bins = 30)

# ExactOneSampleKSTest(DT[DT.SystemProduction .> 0,:SystemProduction],tmp_dist)
# ApproximateOneSampleKSTest(DT[DT.SystemProduction .> 0,:SystemProduction],tmp_dist)

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

# https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/telco/#saving_our_model
# https://juliaai.github.io/MLJ.jl/dev/models/EvoTreeRegressor_EvoTrees/#EvoTreeRegressor_EvoTrees
# https://evovest.github.io/EvoTrees.jl/dev/tutorials/examples-API/
# https://www.juliabloggers.com/using-evotrees-jl-for-time-series-prediction/
# https://iqvia-ml.github.io/LightGBM.jl/dev/functions/#LightGBM.LGBMRegression-Tuple{}
# https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/