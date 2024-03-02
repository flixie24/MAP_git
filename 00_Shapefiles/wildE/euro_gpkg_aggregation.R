# Load the terra package
library(terra)
library(tidyterra)
library(sf)

# Set the path to your GeoPackage file
gpkg_file <- "/Volumes/_wildE_Mapping/00_Shapefiles/grid_europe.gpkg"

# Read the GeoPackage file
europe <- st_read(gpkg_file)

plot(europe)

europe_wildE <- europe %>% filter(Active_YN==1)

plot(europe_wildE)

# Combine all features into a single feature
europe_combined <- st_union(europe_wildE) 

plot(europe_combined)

de <- st_read("/Volumes/_wildE_Mapping/00_Shapefiles/wildE/DEU.gpkg")
#plot(de)

# Transform the europe dataset to have the same CRS as the DEU dataset
europe_transformed <- st_transform(europe_combined, st_crs(de))

# Plot the transformed Europe dataset
plot(europe_transformed)

# Save the transformed dataset to a new GeoPackage file
st_write(europe_transformed, "/Volumes/_wildE_Mapping/00_Shapefiles/wildE/europe_aggregated.gpkg")


################### method 2 using nuts data and terra package

eu <- vect("/Users/shawn/Downloads/NUTS_RG_01M_2021_3035/NUTS_RG_01M_2021_3035.shp") %>% 
  filter(LEVL_CODE==0)
eu <- eu[eu$LEVL_CODE==0]
names(eu)
plot(eu)

eu_aggregated <- aggregate(eu)

plot(eu_aggregated)

writeVector(eu_aggregated, "/Volumes/_wildE_Mapping/00_Shapefiles/wildE/europe_aggregated.gpkg", overwrite=TRUE)

