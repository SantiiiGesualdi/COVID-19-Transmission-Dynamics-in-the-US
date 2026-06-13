using HTTP
using JSON3
using CSV
using DataFrames
using Parquet2
using Downloads: download
using Dates


url      = "https://api.github.com/repos/CSSEGISandData/COVID-19/contents/csse_covid_19_data/csse_covid_19_daily_reports_us?ref=master"
response = HTTP.get(url)

files = JSON3.read(response.body)

# Filter csv files and get download url (dwl_ulr)
dwl_url = [f.download_url for f in files if endswith(f.name, ".csv")]

# Download and read CSVs as a DataFrame list and join them
df_list = [CSV.read(download(u), DataFrame) for u in dwl_url]

df      = vcat(df_list..., cols=:union)

dropmissing!(df, :Last_Update) # Drop rows with missing Last_Update values

date_format = dateformat"yyyy-mm-dd HH:MM:SS"
transform!(df, :Last_Update => ByRow(x -> DateTime(x, date_format)) => :Last_Update)

# Clean up a little the DataFrame
if "Date" in names(df)
    select!(df, Not(:Date))
end

rename!(df, :Confirmed => :Accumulated_Cases, 
            :Deaths => :Accumulated_Deaths, 
            :Province_State => :State)

sort!(df, [:Last_Update, :State])


# Save as Parquet file
out_dir = joinpath(pwd(), "data/cases_wrapped.parquet")
Parquet2.writefile(out_dir, df)