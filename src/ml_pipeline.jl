using Turing, Flux, MLJ, Plots
using DataFrames, Statistics, LinearAlgebra
using Distributions, StatsBase, Random

# Read object in julia format 
DT = load_object("data/pheno.jld2")
Mt = load_object("data/met.jld2")

# Explore metabolites
Mt_details = describe(Mt)
met_names = Mt_details[Mt_details.nmissing .== 0 ,:variable]
cv_variation = 100*std.(eachcol(Mt[:,met_names[2:655]])) ./ mean.(eachcol(Mt[:,met_names[2:655]]))
Mt_cv = DataFrame(names = met_names[2:655],cv = cv_variation)
sort(Mt_cv,:cv, rev = true)

# Join dataframes
DT = innerjoin(DT,Mt[:,met_names], on = :sample_id => :sample)

# Remove lines with missing data 
dropmissing!(DT)
unique!(DT)
describe(DT)
schema(DT)

train, test = partition(collect(eachindex(DT.sample_id)), 0.8, shuffle=true, rng=438)
X = DT[:,vcat(2:5,7:10)]
y = DT.fev1_post;

# Machine learning pipeline
RandomForest = MLJ.@load RandomForestRegressor pkg=DecisionTree
LassoRegressor = MLJ.@load LassoRegressor pkg=MLJLinearModels
rf_model = RandomForest();
la_model = LassoRegressor(solver = MLJLinearModels.ProxGrad(max_iter = 10000))

pipe_transformer = (X -> coerce!(X, :gender=>OrderedFactor, 
                                     :race=>OrderedFactor,
                                     :smoking_status=>OrderedFactor, 
                                     :center=>OrderedFactor)) |> ContinuousEncoder()

rf_leaf = range(rf_model, :min_samples_leaf, lower = 5, upper = 20)
rf_mtry = range(rf_model, :n_subfeatures, lower = 1, upper = 8)
rf_tree = range(rf_model, :n_trees, values = [100, 250, 500, 1000, 2500])
rf_tm = TunedModel(model = rf_model, 
                   tuning = Grid(resolution = 10), # RandomSearch()
                   resampling = CV(nfolds = 5, rng = 123), 
                   ranges = [rf_leaf, rf_mtry, rf_tree],
                   measure = rms)
                   
pipe = pipe_transformer |> rf_tm
pipe_shrinkage = pipe_transformer |> la_model

rf_mtm = machine(pipe_sh, X, y)
fit!(rf_mtm, rows = train);

lm_lasso = machine(pipe_shrinkage, X, y)
fit!(lm_lasso, rows = train)
evaluate!(lm_lasso, resampling = CV(nfolds=10, rng=1234), measure = [rsquared, rmse])

predictions_rf = MLJ.predict(rf_mtm, rows=test);
root_mean_squared_error(y[test],predictions_rf)

predictions_sh = MLJ.predict(lm_lasso, rows=test);
root_mean_squared_error(y[test],predictions_sh)

#scatter(y[test],predictions_rf)

# Stacking models
DecisionTreeRegressor = MLJ.@load DecisionTreeRegressor pkg=DecisionTree
EvoTreeRegressor = MLJ.@load EvoTreeRegressor
XGBoostRegressor = MLJ.@load XGBoostRegressor
KNNRegressor = MLJ.@load KNNRegressor pkg=NearestNeighborModels
LinearRegressor = MLJ.@load LinearRegressor pkg=MLJLinearModels

stack = Stack(;metalearner = LinearRegressor(),
                resampling = CV(),
                measures = rmse,
                constant = ConstantRegressor(),
                tree_2 = DecisionTreeRegressor(max_depth = 2),
                tree_3 = DecisionTreeRegressor(max_depth = 3),
                evo = EvoTreeRegressor(),
                knn = KNNRegressor(),
                xgb = XGBoostRegressor())
stack_pipe = pipe_transformer |> stack
mach = machine(stack_pipe, X, y)
fit!(mach, rows = train);
evaluate!(mach; resampling = CV(nfolds=10, rng=1234), measure = [rsquared, rmse])

tmp = report(mach)
tmp.deterministic_stack.cv_report
