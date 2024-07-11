Solar energy forecast
================

## Import libraries

Import libraries for data cleaning, statistics, machine learning and
visualization

``` julia
using Dates, DataFrames, Plots, StatsPlots, Statistics, Distributions
using StatsBase, HypothesisTests, LinearAlgebra, Random, MLJ, CSV, CategoricalArrays 
```

## Dataset

Our data contains information on these key factors:

- **timedate**: date and hour of each datapoint
- **WindSpeed**: wind speed in Km/h
- **Sunshine**: minutes per hours that sun is not cover by clouds
  (scaled 0-60)
- **AirPressure**: athmosphere pressure in hPa
- **Radiation**: solar radiation W/m<sup>2</sup>
- **AirTemperature**: air temperature in Celsius
- **RelativeAirHumidity**: relative humidity (scaled 0-100)
- **SystemProduction**: system production kWh

``` julia
DT = CSV.read("C:/Users/nicol/Documents/solar_1/Solar Power Plant Data.csv", DataFrame);
show(describe(DT),allcols = true)
```

    8×7 DataFrame
     Row │ variable             mean     min               median  max               nmissing  eltype   
         │ Symbol               Union…   Any               Union…  Any               Int64     DataType 
    ─────┼──────────────────────────────────────────────────────────────────────────────────────────────
       1 │ Date-Hour(NMT)                01.01.2017-00:00          31.12.2017-23:00         0  String31
       2 │ WindSpeed            2.63982  0.0               2.3     10.9                     0  Float64
       3 │ Sunshine             11.1805  0                 0.0     60                       0  Int64
       4 │ AirPressure          1010.36  965.9             1011.0  1047.3                   0  Float64
       5 │ Radiation            97.5385  -9.3              -1.4    899.7                    0  Float64
       6 │ AirTemperature       6.97889  -12.4             6.4     27.1                     0  Float64
       7 │ RelativeAirHumidity  76.7194  13                82.0    100                      0  Int64
       8 │ SystemProduction     684.746  0.0               0.0     7701.0                   0  Float64
