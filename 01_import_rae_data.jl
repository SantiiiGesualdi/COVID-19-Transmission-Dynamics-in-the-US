using HTTP
using JSON3
using CSV
using DataFrames
using Parquet2
using Downloads: download


url      = "https://api.github.com/repos/CSSEGISandData/COVID-19/contents/csse_covid_19_data/csse_covid_19_daily_reports_us?ref=master"
response = HTTP.get(url)

files = JSON3.read(response.body)

# Filter csv files and get download url (dwl_ulr)
dwl_url = [f.download_url for f in files if endswith(f.name, ".csv")]

# Download and read CSVs as a DataFrame list and join them
df_list = [CSV.read(download(u), DataFrame) for u in dwl_url]

df      = vcat(df_list..., cols=:union)


# Clean up a little the DataFrame
if "Date" in names(df)
    select!(df, Not(:Date))
end

rename!(df, :Confirmed => :Accumulated_Cases, 
             :Deaths => :Accumulated_Deaths, 
             :Province_State => :State)

sort!(df, [:Last_Update, :State])

cols = names(df)

filter!(x -> x != "Last_Update", cols)           
state_idx = findfirst(==("State"), cols)         
insert!(cols, state_idx + 1, "Last_Update")      
select!(df, cols)                               


# Save as Parquet file
out_dir = "data/"
mkpath(out_dir)

Parquet2.writefile(joinpath(out_dir, "Cases_Wrapped.parquet"), df)