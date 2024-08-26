# Load libraries and import  dataset
using Dates, DataFrames, Plots, StatsPlots, Statistics, Distributions, ConformalPrediction
using StatsBase, HypothesisTests, LinearAlgebra, Random, MLJ, CSV, CategoricalArrays 

DT = CSV.read("C:/Users/nicol/Documents/solar_1/Solar Power Plant Data.csv", DataFrame);

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

suspect_day = dt[(dt.Radiation_mean .> 40) .&& (dt.SystemProduction_sum .< 1800),:date_real]
dt.is_suspect = zeros(size(dt,1))
dt[(dt.Radiation_mean .> 40) .&& (dt.SystemProduction_sum .< 1800),:is_suspect] .= 1.0;
zero_prod = dt[dt.SystemProduction_sum .== 0,:]

dt = dt[dt.is_suspect .== 0.0,:]
plot(dt.date_real,dt.SystemProduction_sum)
filter!([:date_real, :SystemProduction_sum, :Radiation_mean] => (x,y,z) -> x ∉ Ref(suspect_day) && 
        !(y == 0 && z > 40) && y > 0, dt)

function cyclical_encoder(df::DataFrame, columns::Union{Array, Symbol}, max_val::Union{Array, Int} )
    for (column, max) in zip(columns, max_val)        
        df[:, Symbol(string(column) * "_sin")] = sin.(2*pi*df[:, column]/max)
        df[:, Symbol(string(column) * "_cos")] = cos.(2*pi*df[:, column]/max)
    end
    return df
end

# Parse properly all dates and hours
dt.day = Dates.day.(dt.date_real)
dt.month = Dates.month.(dt.date_real)

cyclical_encoder(dt, ["day","month"], [31,12])

train, test = (collect(1:202),collect(203:302));
X = dt[:,vcat(3:8,13:16)];
y = dt[:,:SystemProduction_sum];

EvoTreeRegressor = MLJ.@load EvoTreeRegressor pkg=EvoTrees
et_regressor = EvoTreeRegressor(nbins = 32, max_depth = 10, nrounds = 200)

conf_model = conformal_model(et_regressor; method=:jackknife_plus_ab_minmax, coverage=0.9)
mach = machine(conf_model, X, y)
fit!(mach, rows=train)

evaluate!(mach, resampling = CV(nfolds=5, rng=1234), 
          repeats=5, measure = [emp_coverage, ssc])

y_pred_interval = MLJ.predict(conf_model, mach.fitresult, X[test,:])
lb = [ minimum(tuple_data) for tuple_data in y_pred_interval]
ub = [ maximum(tuple_data) for tuple_data in y_pred_interval]
y_pred = [mean(tuple_data) for tuple_data in y_pred_interval]

plot(collect(1:100),y_pred)
plot!(collect(1:100),y[test])
plot!(collect(1:100), lb, fillrange = ub, 
      fillalpha = 0.2,color=:lake, linewidth=0, framestyle=:box)