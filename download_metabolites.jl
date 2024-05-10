using JSON, HTTP, DataFrames, CSV

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

# resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/study_id/ST002089/data/");
# https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN000001/datatable/
resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN003410/mwtab");
str = String(resp.body);
jobj = JSON.Parser.parse(str);

keys_metabolite = string.(keys(jobj))
jobj["MS_METABOLITE_DATA"]["Metabolites"]
codebook_met = vcat(DataFrame.(jobj["MS_METABOLITE_DATA"]["Metabolites"])...)
allowmissing!(codebook_met)
for col in eachcol(codebook_met)
    replace!(col, "NA" => missing)
end

data_met = vcat(DataFrame.(jobj["MS_METABOLITE_DATA"]["Data"])...)
allowmissing!(data_met)
for col in eachcol(data_met)
    replace!(col, "NA" => missing)
end
disallowmissing!(data_met,:Metabolite)
data_met = permutedims(data_met,1126)

# dataset = CSV.read(download("https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN003410/datatable/"), DataFrame)

resp = HTTP.get("https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN003411/mwtab");
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
data_met = permutedims(data_met,1126)