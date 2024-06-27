using JSON, HTTP, DataFrames, CSV, Missings, JLD2 

resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/study_id/ST002089/allfactors/");
str = String(resp.body);
jobj = JSON.Parser.parse(str);

keys_factor = string.(keys(jobj))

tmp = vcat(jobj[keys_factor[1]]["local_sample_id"],
           strip.(split(jobj[keys_factor[1]]["factors"],('|',':'))[collect(2:2:18)]))
for i in 2:length(keys_factor)
    a1 = vcat(jobj[keys_factor[i]]["local_sample_id"],
              strip.(split(jobj[keys_factor[i]]["factors"],('|',':'))[collect(2:2:18)]))
    tmp = hcat(tmp,a1)
end

DT = permutedims(tmp)
DT[DT[:,10] .== "-",10] .= "0.0"
DT = convert(Matrix{Union{AbstractString,Missing}},DT)
DT[DT .== "NA"] .= missing
#=for i in 1:length(keys_factor), j in 1:10
    if DT[i,j] == "NA"
        DT[i,j] = missing
    else
        DT[i,j] = DT[i,j]
    end
end=#

dt_names = ["sample_id","gender","race","smoking_status","center","fev1_post","adj_density_mesa",
            "age_visit","bmi","pack_years"]
DT = DataFrames.DataFrame(DT,:auto)
rename!(DT,dt_names)
transform!(DT, dt_names[6:10] .=> ByRow(x -> Missings.passmissing(parse).(Float64, x)), 
           renamecols=false)

# resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/study_id/ST002089/data/");
# https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN000001/datatable/
resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN003412/mwtab");
str = String(resp.body);
jobj = JSON.Parser.parse(str);

keys_metabolite = string.(keys(jobj))
jobj["MS_METABOLITE_DATA"]["Metabolites"]
codebook_met = vcat(DataFrame.(jobj["MS_METABOLITE_DATA"]["Metabolites"])...)
allowmissing!(codebook_met)
for col in eachcol(codebook_met)
    replace!(col, "NA" => missing)
end
codebook_met.label = string.("met_",collect(1:size(codebook_met,1)))

data_met = vcat(DataFrame.(jobj["MS_METABOLITE_DATA"]["Data"])...)
allowmissing!(data_met)
for col in eachcol(data_met)
    replace!(col, "NA" => missing)
end
disallowmissing!(data_met,:Metabolite)
data_met = permutedims(data_met,1126)
rename!(data_met,vcat("sample",string.("met_",collect(1:size(codebook_met,1)))))
transform!(data_met, names(data_met[:,2:end]) .=> ByRow(x -> Missings.passmissing(parse).(Float64, x)), 
           renamecols=false)

# Save object in julia format 
# save_object("data/pheno.jld2", DT)
# save_object("data/codebook.jld2", codebook_met)
# save_object("data/met.jld2", data_met)

# Alternative use to download table format
# url_table = "https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN003410/datatable/"
# dataset = CSV.read(download(url_table), DataFrame)

#=resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN003411/mwtab");
str = String(resp.body);
jobj = JSON.Parser.parse(str);

keys_metabolite = string.(keys(jobj))
jobj["MS_METABOLITE_DATA"]["Metabolites"]
codebook_met = vcat(DataFrame.(jobj["MS_METABOLITE_DATA"]["Metabolites"])...)
allowmissing!(codebook_met)
for col in eachcol(codebook_met)
    replace!(col, "NA" => missing, "" => missing)
end

data_met = vcat(DataFrame.(jobj["MS_METABOLITE_DATA"]["Data"])...)
allowmissing!(data_met)
for col in eachcol(data_met)
    replace!(col, "NA" => missing, "" => missing)
end
disallowmissing!(data_met,:Metabolite)
data_met = permutedims(data_met,1126) =#

# https://github.com/JuliaStats/NMF.jl --> Non-negative Matrix Factorization

# Check missing data and outliers
# missing_met = describe(data_met, :nmissing)
# missing_met[missing_met.nmissing .== 0,:]

#=https://www.dati.lombardia.it/resource/beda-kb7b.csv?$limit=5000000
url_table = raw"https://www.dati.lombardia.it/resource/beda-kb7b.csv?$limit=5000000"
dataset = CSV.read(download(url_table), DataFrame)

resp = HTTP.get(raw"https://www.dati.lombardia.it/resource/g2hp-ar79.json?$limit=50000000");
str = String(resp.body);
jobj = JSON.Parser.parse(str); =#

#=url_table = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63087/suppl/GSE63087_normalized_without_technical_replicates.txt.gz"
download(url_table)=#







