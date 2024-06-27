using Turing, Flux, MLJ
using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random

# Read object in julia format 
DT = load_object("data/pheno.jld2")

# Remove lines with missing data 
dropmissing!(DT)

describe(DT)
schema(DT)
coerce!(DT, :gender=>OrderedFactor, :race=>OrderedFactor, 
            :smoking_status=>OrderedFactor, :center=>OrderedFactor)

train, test = partition(collect(eachindex(DT.sample_id)), 0.8, shuffle=true, rng=418)
X = DT[:,vcat(2:5,7:10)]
y = DT.fev1_post;

# Machine learning pipeline
RandomForest = MLJ.@load RandomForestRegressor pkg=DecisionTree
rf_model = RandomForest();

rf_leaf = range(rf_model, :min_samples_leaf, lower = 10, upper = 30)
rf_mtry = range(rf_model, :n_subfeatures, lower = 1, upper = 8)
rf_tree = range(rf_model, :n_trees, values = [100, 250, 500, 1000, 2500])
rf_tm = TunedModel(model = rf_model, 
                   tuning = Grid(resolution = 10), # RandomSearch()
                   resampling = CV(nfolds = 5, rng = 123), 
                   ranges = [rf_leaf,rf_mtry,rf_tree],
                   measure = rms)
rf_mtm = machine(rf_tm, X, y)
fit!(rf_mtm, rows = train);

predictions_rf = MLJ.predict(rf_mtm, rows=test);
root_mean_squared_error(y[test],predictions_rf)