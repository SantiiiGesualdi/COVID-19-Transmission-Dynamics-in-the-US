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
data_list = [CSV.read(download(u), DataFrame) for u in dwl_url]

data      = vcat(data_list..., cols=:union)

dropmissing!(data, :Last_Update) # Drop rows with missing Last_Update values

date_format = dateformat"yyyy-mm-dd HH:MM:SS"
transform!(data, :Last_Update => ByRow(x -> DateTime(x, date_format)) => :Last_Update)

# Clean up a little the DataFrame
if "Date" in names(data)
    select!(data, Not(:Date))
end

rename!(data, :Confirmed => :Accumulated_Cases, 
            :Deaths => :Accumulated_Deaths, 
            :Province_State => :State)

sort!(data, [:Last_Update, :State])


select!(data, :State, :Last_Update, :Accumulated_Cases, 
              :Accumulated_Deaths, :Active, :Recovered,
              :People_Hospitalized)

lag0(x) = vcat(0, x[1:end-1]) 

gdf_state = groupby(data, :State)

# Calculate the New_Cases and New_Deaths columns
transform!(gdf_state, 
    :Accumulated_Cases => (x -> x .- lag0(x)) => :New_Cases,
    :Accumulated_Deaths => (x -> x .- lag0(x)) => :New_Deaths
)

# Detect the upload period
transform!(gdf_state, :New_Cases => (x -> x .!= 0) => :IS_Report)
transform!(gdf_state, :IS_Report => cumsum => :ID_Period)

# Create the new incidents columns
gdf_period = groupby(data, [:State, :ID_Period])

df = combine(gdf_period,
    :Last_Update => minimum => :Start_Date,
    :Last_Update => maximum => :Report_Date,
    nrow => :Days_In_Period,         
    :New_Deaths => sum => :New_Deaths,
    :New_Cases => sum => :New_Cases,
    :Accumulated_Cases => maximum => :Accumulated_Cases,
    :Accumulated_Deaths => maximum => :Accumulated_Deaths,
    :Active => maximum => :Active,
    :Recovered => maximum => :Recovered,
    :People_Hospitalized => maximum => :People_Hospitalized
)

transform!(df, 
    [:Accumulated_Deaths, :Accumulated_Cases] => ByRow((d, c) -> c == 0 ? 0.0 : d / c) => :Accumulated_Deaths_Rate,
    [:New_Deaths, :New_Cases] => ByRow((d, c) -> c == 0 ? 0.0 : d / c) => :New_Deaths_Rate
)


sort!(df, [:State, :Start_Date])

# Save cleaned dataset as Parquet file
mkpath("data")
out_dir = joinpath(pwd(), "data/cases_deaths.parquet")
Parquet2.writefile(out_dir, df)
