from osgeo import ogr, osr, gdal
import geemap, ee
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  
import math
from scipy.stats import norm, gamma, f, chi2
import concurrent.futures
import random
import os, shutil


# ############################################## #
# #### General QOL Tools #### #

def DeleteAllItemsInFolder(path):
    print(f"Deleting all items in folder: {path}")
    # Loop through items in folder, and delete them one by one
    itemList = os.listdir(path)
    if not itemList:
        print(f"Folder {path} is already empty.")
        return
    
    for item in itemList:
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            print(f"Deleting file: {item_path}")
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            print(f"Deleting directory: {item_path}")
            shutil.rmtree(item_path)
    print("All items in {path} have been deleted.")

    # Check if folder is empty after deletion
    itemList = os.listdir(path)
    if not itemList:
        print(f"Folder {path} is now empty.")


# ############################################## #
# #### TOOLS TO WORK WITH SPATIAL DATA #### #
        
def FC_to_pandas(fc):
  init_flag_pd = 0
  feats = fc['features']
  for feat in feats:
    prop = feat['properties']
    if init_flag_pd == 0:
      out_pd = pd.DataFrame(prop, index=[0])
      init_flag_pd = 1
    else:
      out_pd = pd.concat([out_pd, pd.DataFrame(prop, index=[0])])
  return out_pd

# Temporary storage for functions--> move them later to collection
def ProjSite_ToeeGeom(path):
    def my_envelope(geom):
        (minX, maxX, minY, maxY) = geom.GetEnvelope()

        # Create ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(minX, minY)
        ring.AddPoint(maxX, minY)
        ring.AddPoint(maxX, maxY)
        ring.AddPoint(minX, maxY)
        ring.AddPoint(minX, minY)

        # Create polygon
        poly_envelope = ogr.Geometry(ogr.wkbPolygon)
        poly_envelope.AddGeometry(ring)
        return poly_envelope
    # Open file, get first (hopefully only) feature
    ds       = ogr.Open(path)
    lyr      = ds.GetLayer()
    cnt      = lyr.GetFeatureCount()
    if cnt > 1:
        print("Shapefile has more than one feature, taking the first one...")
    feat     = lyr.GetNextFeature()
    geom     = feat.geometry()
    envelope = my_envelope(geom)
    cHull    = geom.ConvexHull()
    lyr.ResetReading()
    # Convert to EPSG:4326
    from_sr = lyr.GetSpatialRef()
    from_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_sr   = osr.SpatialReference()
    to_sr.ImportFromEPSG(4326)
    to_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    cs      = osr.CoordinateTransformation(from_sr, to_sr)
    envelope.Transform(cs)
    cHull.Transform(cs)
    # Convert to ee Geometry
    # Envelope
    envelopeJSON    = json.loads(envelope.ExportToJson())
    envelopeCoord   = [c[0:2] for c in envelopeJSON['coordinates'][0]]
    eeGeom_envelope = ee.Geometry.Polygon(envelopeCoord)
    # Convex Hull
    chullJSON       = json.loads(cHull.ExportToJson())
    chullCoord      = [c[0:2] for c in chullJSON['coordinates'][0]]
    eeGeom_chull    = ee.Geometry.Polygon(chullCoord)
    return eeGeom_envelope, eeGeom_chull


# Build the grid data layer
def BuildGrid(in_lyr, maxGridSize_km=10):
    # Calculate the distance of max Extent in x and y direction
    # We have to convert the points into a equal distant coordinate system
    #maxGridSize_km = 40
    # Get extent
    minX, maxX, minY, maxY = in_lyr.GetExtent()
    #print(minX, maxX, minY, maxY)
    # Build UL LR
    ul = ogr.Geometry(ogr.wkbPoint)
    ul.AddPoint(minX, maxY)
    lr = ogr.Geometry(ogr.wkbPoint)
    lr.AddPoint(maxX, minY)
    # Transform the coordinates
    epsg4326    = osr.SpatialReference()
    epsg4326.ImportFromEPSG(4326)
    eckertIV_SR = osr.SpatialReference()
    eckertIV_SR.ImportFromESRI(['PROJCS["ProjWiz_Custom_Eckert_IV",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Eckert_IV"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0],UNIT["Meter",1.0]]'])
    cT_01       = osr.CoordinateTransformation(epsg4326, eckertIV_SR)
    lr.Transform(cT_01)
    ul.Transform(cT_01)
    # Define the distances in km, and number of squares
    xDist = (lr.GetX() - ul.GetX()) / 1000
    yDist = (ul.GetY() - lr.GetY()) / 1000
    xSq   = math.ceil((xDist / maxGridSize_km))
    ySq   = math.ceil((yDist / maxGridSize_km))
    #print(xSq, ySq)
    # Define the StepSize in x and y direction
    xStep = (maxX - minX) / abs(xSq)
    yStep = (maxY - minY) / abs(ySq)
    #print(xStep, yStep)
    # Create a new layer, add ID field
    drvMemV = ogr.GetDriverByName('memory')
    outSHP  = drvMemV.CreateDataSource('')
    lyr_out = outSHP.CreateLayer('', srs=epsg4326, geom_type=ogr.wkbPolygon)
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    lyr_out.CreateField(idField)
    lyrDef  = lyr_out.GetLayerDefn()
    # Loop through the steps
    idNR = 1
    for x in np.arange(minX, maxX, xStep):
        for y in np.arange(minY, maxY, yStep):
            # Create feature
            feat = ogr.Feature(lyrDef)
            feat.SetField('id', idNR)
            idNR += 1
            # Build geometry
            square = ogr.Geometry(ogr.wkbLinearRing)
            square.AddPoint(x, y)
            square.AddPoint(x, y+yStep)
            square.AddPoint(x+xStep, y+yStep)
            square.AddPoint(x+xStep, y)
            square.AddPoint(x, y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(square)
            # Set geoemtry into feature
            feat.SetGeometry(poly)
            lyr_out.CreateFeature(feat)
    return outSHP


# Remove doublecounts in the collections
def remove_double_counts(collection):
    # function to remove double counts from path overlap areas
    def add_nn(image):
        start       = ee.Date.fromYMD(image.date().get('year'), image.date().get('month'),
                                image.date().get('day')).update(hour=0, minute=0, second=0)
        end         = ee.Date.fromYMD(image.date().get('year'), image.date().get('month'), image.date().get('day')).update(
                      hour=23, minute=59, second=59)
        overlapping = collection.filterDate(start, end).filterBounds(image.geometry())
        nn          = overlapping.filterMetadata('WRS_ROW', 'equals', ee.Number(image.get('WRS_ROW')).subtract(1)).size()
        return image.set('nn', nn)

    collection_nn = collection.map(add_nn)
    has_nn = collection_nn.filterMetadata('nn', 'greater_than', 0)
    has_no_nn = ee.ImageCollection(ee.Join.inverted().apply(collection, has_nn,
                                                            ee.Filter.equals(leftField='LANDSAT_ID',
                                                                            rightField='LANDSAT_ID')))
    def mask_overlap(image):
        start = ee.Date.fromYMD(image.date().get('year'), image.date().get('month'),
                                image.date().get('day')).update(hour=0, minute=0, second=0)
        end = ee.Date.fromYMD(image.date().get('year'), image.date().get('month'), image.date().get('day')).update(
            hour=23, minute=59, second=59)
        overlapping = collection.filterDate(start, end).filterBounds(image.geometry())
        nn = ee.Image(
            overlapping.filterMetadata('WRS_ROW', 'equals', ee.Number(image.get('WRS_ROW')).subtract(1)).first())
        newmask = image.mask().where(nn.mask(), 0)
        return image.updateMask(newmask)
    has_nn_masked = ee.ImageCollection(has_nn.map(mask_overlap))
    out = ee.ImageCollection(has_nn_masked.merge(has_no_nn).copyProperties(collection))
    return out

def Seasonal_STMs_C2(year, roi, startMonth, endMonth, filter_double_counts=False, reducer="all"):
# Define reducers
    mean        = ee.Reducer.mean().unweighted()
    sd          = ee.Reducer.stdDev().unweighted()
    percentiles = ee.Reducer.percentile([10, 25, 50,75, 90]).unweighted()
    allMetrics  = mean.combine(sd, sharedInputs=True).combine(percentiles, sharedInputs=True)

    yr_ee = ee.Number(year)
    sM_ee = ee.Number(startMonth)
    eM_ee = ee.Number(endMonth)

    yr_start_ee = ee.Algorithms.If(yr_ee.lte(ee.Number(1986)), yr_ee.subtract(1), yr_ee)
    yr_end_ee   = ee.Algorithms.If(yr_ee.lte(ee.Number(1986)), yr_ee.add(1), yr_ee)

    lastDay     = ee.Algorithms.If(ee.List([4, 6, 9, 11]).contains(eM_ee), ee.Number(30),ee.Number(0))
    lastDay     = ee.Algorithms.If(ee.List([1, 3, 5, 7, 8, 10, 12]).contains(eM_ee), ee.Number(31), lastDay)
    lastDay     = ee.Algorithms.If(eM_ee.eq(2), ee.Number(28), lastDay)

    start = ee.Date.fromYMD(yr_start_ee, sM_ee, ee.Number(1))
    end   = ee.Date.fromYMD(yr_end_ee, eM_ee, lastDay)
    
    def ScaleMaskIndices_l89(img):
        ## Scale the bands
        refl   = img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)
        img_SR = ee.Image(refl).addBands(img.select(['QA_PIXEL']))
        ## mask cloud
        # Get the QA band and apply the bits
        qa = ee.Image(img_SR).select(['QA_PIXEL'])
        dilatedCloud = qa.bitwiseAnd(1 << 1)
        cirrusMask   = qa.bitwiseAnd(1 << 2)
        cloud        = qa.bitwiseAnd(1 << 3)
        cloudShadow  = qa.bitwiseAnd(1 << 4)
        snow         = qa.bitwiseAnd(1 << 5)
        water        = qa.bitwiseAnd(1 << 7)
        # Apply the bits
        clear = dilatedCloud.Or(cloud).Or(cloudShadow).eq(0)
        cfmask = (clear.eq(0).where(water.gt(0), 1)
              .where(snow.gt(0), 3)
              .where(dilatedCloud.Or(cloud).gt(0), 4)
              .where(cloudShadow.gt(0), 2)
              .rename('cfmask'))
        img_SR_cl = ee.Image(img_SR).select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).mask(cfmask.lte(2))
        # Hamonize to L45 and 7
        #interc = ee.Image.constant([0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116])
        #slope = ee.Image.constant([0.885, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165])
        #img_SR_cl = img_SR_cl.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(slope).add(interc)

        #slope = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]) # create an image of slopes per band for L8 TO L7 regression line - David Roy
        #intercept = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]) # create an image of y-intercepts per band for L8 TO L7 regression line - David Roy
        #img_SR_cl = img_SR_cl.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).subtract(intercept.divide(slope))
        # calculate the spectral indices
        ndvi       = img_SR_cl.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        swir_ratio = img_SR_cl.expression('SWIR_2/SWIR_1',{'SWIR_1': img_SR_cl.select('SR_B6'), 'SWIR_2': img_SR_cl.select('SR_B7')}).rename('SWIRratio')
        ndmi       = img_SR_cl.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI')
        nbr        = img_SR_cl.normalizedDifference(['SR_B4', 'SR_B6']).rename('NBR')
        nbr2       = img_SR_cl.normalizedDifference(['SR_B6', 'SR_B7']).rename('NBR2')
        msi        = img_SR_cl.expression('SWIR_1/NIR',{'SWIR_1': img_SR_cl.select('SR_B6'), 'NIR': img_SR_cl.select('SR_B5')}).rename('MSI')
        ndwi       = img_SR_cl.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        evi        = img_SR_cl.expression('2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))', {'NIR': img_SR_cl.select('SR_B5'), 'R': img_SR_cl.select('SR_B4'), 'B': img_SR_cl.select('SR_B2')}).rename('EVI')
        msavi      = img_SR_cl.expression('(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - R)) ) / 2', {'NIR': img_SR_cl.select('SR_B5'), 'R': img_SR_cl.select('SR_B4')}).rename('MSAVI')
        # Calculate tasseled cap
        # coefficients for Landsat surface reflectance (Crist 1985)
        brightness_c = ee.Image([0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863])
        greenness_c  = ee.Image([-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800])
        wetness_c    = ee.Image([0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572])
        brightness   = img_SR_cl.multiply(brightness_c)
        greenness    = img_SR_cl.multiply(greenness_c)
        wetness      = img_SR_cl.multiply(wetness_c)
        brightness   = brightness.reduce(ee.call('Reducer.sum'))
        greenness    = greenness.reduce(ee.call('Reducer.sum'))
        wetness      = wetness.reduce(ee.call('Reducer.sum'))
        tasseled_cap = ee.Image(brightness).addBands(greenness).addBands(wetness).rename(['tcB', 'tcG', 'tcW'])
        # Add all the bands together, rename and then return
        out_img = img_SR_cl.addBands(ndvi).addBands(swir_ratio).addBands(ndmi).addBands(nbr).addBands(nbr2).addBands(msi).addBands(ndwi).addBands(evi).addBands(msavi).addBands(tasseled_cap)\
        .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'SWIRratio', 'NDMI', 'NBR','NBR2', 'MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'],
               ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI',  'SWIRratio', 'NDMI', 'NBR', 'NBR2', 'MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'])\
        .set('YYYYDDD', ee.Date(img.get('system:time_start')).format('YYYYDDD'))
        return out_img
    def ScaleMaskIndices_l457(img):
        ## Scale the bands
        refl = img.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).multiply(0.0000275).add(-0.2)
        img_SR = ee.Image(refl).addBands(img.select(['QA_PIXEL']))
        ## mask cloud
        # Get the QA band and apply the bits
        qa = ee.Image(img_SR).select(['QA_PIXEL'])
        dilatedCloud = qa.bitwiseAnd(1 << 1)
        cirrusMask   = qa.bitwiseAnd(1 << 2)
        cloud        = qa.bitwiseAnd(1 << 3)
        cloudShadow  = qa.bitwiseAnd(1 << 4)
        snow         = qa.bitwiseAnd(1 << 5)
        water        = qa.bitwiseAnd(1 << 7)
        # Apply the bits
        clear = dilatedCloud.Or(cloud).Or(cloudShadow).eq(0)
        cfmask = (clear.eq(0).where(water.gt(0), 1)
              .where(snow.gt(0), 3)
              .where(dilatedCloud.Or(cloud).gt(0), 4)
              .where(cloudShadow.gt(0), 2)
              .rename('cfmask'))
        img_SR_cl = ee.Image(img_SR).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).mask(cfmask.lte(2))  
        ## calculate the spectral indices
        ndvi       = img_SR_cl.normalizedDifference(["SR_B4", "SR_B3"]).rename("NDVI")
        swir_ratio = img_SR_cl.expression('SWIR_2/SWIR_1',{'SWIR_1': img_SR_cl.select('SR_B5'), 'SWIR_2': img_SR_cl.select('SR_B7')}).rename('SWIRratio')
        ndmi       = img_SR_cl.normalizedDifference(['SR_B4', 'SR_B5']).rename('NDMI')
        nbr        = img_SR_cl.normalizedDifference(['SR_B4', 'SR_B7']).rename('NBR')
        nbr2       = img_SR_cl.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR2')
        msi        = img_SR_cl.expression('SWIR_1/NIR',{'SWIR_1': img_SR_cl.select('SR_B5'), 'NIR': img_SR_cl.select('SR_B4')}).rename('MSI')
        ndwi       = img_SR_cl.normalizedDifference(['SR_B2', 'SR_B4']).rename('NDWI')
        evi = img_SR_cl.expression('2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))',{'NIR': img_SR_cl.select('SR_B4'), 'R': img_SR_cl.select('SR_B3'), 'B': img_SR_cl.select('SR_B1')}).rename('EVI')
        msavi = img_SR_cl.expression('(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - R)) ) / 2', {'NIR': img_SR_cl.select('SR_B4'), 'R': img_SR_cl.select('SR_B3')}).rename('MSAVI')
        # Calculate tasseled cap
        # coefficients for Landsat surface reflectance (Crist 1985)
        brightness_c = ee.Image([0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863])
        greenness_c  = ee.Image([-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800])
        wetness_c    = ee.Image([0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572])
        brightness   = img_SR_cl.multiply(brightness_c)
        greenness    = img_SR_cl.multiply(greenness_c)
        wetness      = img_SR_cl.multiply(wetness_c)
        brightness   = brightness.reduce(ee.call('Reducer.sum'))
        greenness    = greenness.reduce(ee.call('Reducer.sum'))
        wetness      = wetness.reduce(ee.call('Reducer.sum'))
        tasseled_cap = ee.Image(brightness).addBands(greenness).addBands(wetness).rename(['tcB', 'tcG', 'tcW'])
        # Add all the bands together, rename and then return
        out_img = img_SR_cl.addBands(ndvi).addBands(swir_ratio).addBands(ndmi).addBands(nbr).addBands(nbr2).addBands(msi).addBands(ndwi).addBands(evi).addBands(msavi).addBands(tasseled_cap)\
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'SWIRratio', 'NDMI', 'NBR', 'NBR2','MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'],
               ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'SWIRratio', 'NDMI', 'NBR', 'NBR2','MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'])\
        .set('YYYYDDD', ee.Date(img.get('system:time_start')).format('YYYYDDD')) 
        return out_img
    
    # Get image collection for Tile from each sensor across the defined time range  
    l4   = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2").filterBounds(roi).filter(ee.Filter.date(start, end))
    l5   = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(roi).filter(ee.Filter.date(start, end))
    l7   = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(roi).filter(ee.Filter.date(start, end))
    l8   = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filter(ee.Filter.date(start, end)) 
    l9   = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filter(ee.Filter.date(start, end))  
    
    # Apply the remove double counts function
    if filter_double_counts:
        l4   = remove_double_counts(l4)
        l5   = remove_double_counts(l5)
        l7   = remove_double_counts(l7)
        l8   = remove_double_counts(l8)
        l9   = remove_double_counts(l9)
        
    # Scale the indicies and apply QA masks 
    l4   = l4.map(ScaleMaskIndices_l457)
    l5   = l5.map(ScaleMaskIndices_l457)
    l7   = l7.map(ScaleMaskIndices_l457)
    l8   = l8.map(ScaleMaskIndices_l89)
    l9   = l9.map(ScaleMaskIndices_l89)
    
    #  return merged image collection 
    lALL = ee.ImageCollection(l4.merge(l5).merge(l7).merge(l8).merge(l9))
# Apply the reducers   
    # Apply the reducers based on the 'reducer' argument
    if reducer == "mean":
        stm = lALL.reduce(mean)
    elif reducer == "all":
        stm = lALL.reduce(allMetrics)
    else:
        raise ValueError("Invalid reducer option. Use 'mean' or 'all'.")
    
# When we have the bands extracted, rename them sequentially
    old_bn  = stm.bandNames()
    #print(old_bn.getInfo())
    bandseq = ee.List.sequence(1, old_bn.size())
    def create_bandnames(i):
        return ee.String('v').cat(ee.Number(i).format('%03d'))
    new_bn = bandseq.map(create_bandnames)
    # preserve human readable format by commenting out renaming of bands 
    #stm = stm.select(old_bn, new_bn)
    return stm

def ScaleMaskIndices_l89(img):
    ## Scale the bands
    refl   = img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)
    img_SR = ee.Image(refl).addBands(img.select(['QA_PIXEL']))
    ## mask cloud
    # Get the QA band and apply the bits
    qa = ee.Image(img_SR).select(['QA_PIXEL'])
    dilatedCloud = qa.bitwiseAnd(1 << 1)
    cirrusMask   = qa.bitwiseAnd(1 << 2)
    cloud        = qa.bitwiseAnd(1 << 3)
    cloudShadow  = qa.bitwiseAnd(1 << 4)
    snow         = qa.bitwiseAnd(1 << 5)
    water        = qa.bitwiseAnd(1 << 7)
    # Apply the bits
    clear = dilatedCloud.Or(cloud).Or(cloudShadow).eq(0)
    cfmask = (clear.eq(0).where(water.gt(0), 1)
            .where(snow.gt(0), 3)
            .where(dilatedCloud.Or(cloud).gt(0), 4)
            .where(cloudShadow.gt(0), 2)
            .rename('cfmask'))
    img_SR_cl = ee.Image(img_SR).select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).mask(cfmask.lte(2))
    # Hamonize to L45 and 7
    #interc = ee.Image.constant([0.0183, 0.0123, 0.0123, 0.0448, 0.0306, 0.0116])
    #slope = ee.Image.constant([0.885, 0.9317, 0.9372, 0.8339, 0.8639, 0.9165])
    #img_SR_cl = img_SR_cl.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(slope).add(interc)

    #slope = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]) # create an image of slopes per band for L8 TO L7 regression line - David Roy
    #intercept = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]) # create an image of y-intercepts per band for L8 TO L7 regression line - David Roy
    #img_SR_cl = img_SR_cl.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).subtract(intercept.divide(slope))
    # calculate the spectral indices
    ndvi       = img_SR_cl.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    swir_ratio = img_SR_cl.expression('SWIR_2/SWIR_1',{'SWIR_1': img_SR_cl.select('SR_B6'), 'SWIR_2': img_SR_cl.select('SR_B7')}).rename('SWIRratio')
    ndmi       = img_SR_cl.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI')
    nbr        = img_SR_cl.normalizedDifference(['SR_B4', 'SR_B6']).rename('NBR')
    nbr2       = img_SR_cl.normalizedDifference(['SR_B6', 'SR_B7']).rename('NBR2')
    msi        = img_SR_cl.expression('SWIR_1/NIR',{'SWIR_1': img_SR_cl.select('SR_B6'), 'NIR': img_SR_cl.select('SR_B5')}).rename('MSI')
    ndwi       = img_SR_cl.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    evi        = img_SR_cl.expression('2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))', {'NIR': img_SR_cl.select('SR_B5'), 'R': img_SR_cl.select('SR_B4'), 'B': img_SR_cl.select('SR_B2')}).rename('EVI')
    msavi      = img_SR_cl.expression('(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - R)) ) / 2', {'NIR': img_SR_cl.select('SR_B5'), 'R': img_SR_cl.select('SR_B4')}).rename('MSAVI')
    # Calculate tasseled cap
    # coefficients for Landsat surface reflectance (Crist 1985)
    brightness_c = ee.Image([0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863])
    greenness_c  = ee.Image([-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800])
    wetness_c    = ee.Image([0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572])
    brightness   = img_SR_cl.multiply(brightness_c)
    greenness    = img_SR_cl.multiply(greenness_c)
    wetness      = img_SR_cl.multiply(wetness_c)
    brightness   = brightness.reduce(ee.call('Reducer.sum'))
    greenness    = greenness.reduce(ee.call('Reducer.sum'))
    wetness      = wetness.reduce(ee.call('Reducer.sum'))
    tasseled_cap = ee.Image(brightness).addBands(greenness).addBands(wetness).rename(['tcB', 'tcG', 'tcW'])
    # Add all the bands together, rename and then return
    out_img = img_SR_cl.addBands(ndvi).addBands(swir_ratio).addBands(ndmi).addBands(nbr).addBands(nbr2).addBands(msi).addBands(ndwi).addBands(evi).addBands(msavi).addBands(tasseled_cap)\
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'SWIRratio', 'NDMI', 'NBR','NBR2', 'MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'],
            ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI',  'SWIRratio', 'NDMI', 'NBR', 'NBR2', 'MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'])\
    .set('YYYYDDD', ee.Date(img.get('system:time_start')).format('YYYYDDD'))
    return out_img
def ScaleMaskIndices_l457(img):
    ## Scale the bands
    refl = img.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).multiply(0.0000275).add(-0.2)
    img_SR = ee.Image(refl).addBands(img.select(['QA_PIXEL']))
    ## mask cloud
    # Get the QA band and apply the bits
    qa = ee.Image(img_SR).select(['QA_PIXEL'])
    dilatedCloud = qa.bitwiseAnd(1 << 1)
    cirrusMask   = qa.bitwiseAnd(1 << 2)
    cloud        = qa.bitwiseAnd(1 << 3)
    cloudShadow  = qa.bitwiseAnd(1 << 4)
    snow         = qa.bitwiseAnd(1 << 5)
    water        = qa.bitwiseAnd(1 << 7)
    # Apply the bits
    clear = dilatedCloud.Or(cloud).Or(cloudShadow).eq(0)
    cfmask = (clear.eq(0).where(water.gt(0), 1)
            .where(snow.gt(0), 3)
            .where(dilatedCloud.Or(cloud).gt(0), 4)
            .where(cloudShadow.gt(0), 2)
            .rename('cfmask'))
    img_SR_cl = ee.Image(img_SR).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']).mask(cfmask.lte(2))  
    ## calculate the spectral indices
    ndvi       = img_SR_cl.normalizedDifference(["SR_B4", "SR_B3"]).rename("NDVI")
    swir_ratio = img_SR_cl.expression('SWIR_2/SWIR_1',{'SWIR_1': img_SR_cl.select('SR_B5'), 'SWIR_2': img_SR_cl.select('SR_B7')}).rename('SWIRratio')
    ndmi       = img_SR_cl.normalizedDifference(['SR_B4', 'SR_B5']).rename('NDMI')
    nbr        = img_SR_cl.normalizedDifference(['SR_B4', 'SR_B7']).rename('NBR')
    nbr2       = img_SR_cl.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR2')
    msi        = img_SR_cl.expression('SWIR_1/NIR',{'SWIR_1': img_SR_cl.select('SR_B5'), 'NIR': img_SR_cl.select('SR_B4')}).rename('MSI')
    ndwi       = img_SR_cl.normalizedDifference(['SR_B2', 'SR_B4']).rename('NDWI')
    evi = img_SR_cl.expression('2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))',{'NIR': img_SR_cl.select('SR_B4'), 'R': img_SR_cl.select('SR_B3'), 'B': img_SR_cl.select('SR_B1')}).rename('EVI')
    msavi = img_SR_cl.expression('(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - R)) ) / 2', {'NIR': img_SR_cl.select('SR_B4'), 'R': img_SR_cl.select('SR_B3')}).rename('MSAVI')
    # Calculate tasseled cap
    # coefficients for Landsat surface reflectance (Crist 1985)
    brightness_c = ee.Image([0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863])
    greenness_c  = ee.Image([-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800])
    wetness_c    = ee.Image([0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572])
    brightness   = img_SR_cl.multiply(brightness_c)
    greenness    = img_SR_cl.multiply(greenness_c)
    wetness      = img_SR_cl.multiply(wetness_c)
    brightness   = brightness.reduce(ee.call('Reducer.sum'))
    greenness    = greenness.reduce(ee.call('Reducer.sum'))
    wetness      = wetness.reduce(ee.call('Reducer.sum'))
    tasseled_cap = ee.Image(brightness).addBands(greenness).addBands(wetness).rename(['tcB', 'tcG', 'tcW'])
    # Add all the bands together, rename and then return
    out_img = img_SR_cl.addBands(ndvi).addBands(swir_ratio).addBands(ndmi).addBands(nbr).addBands(nbr2).addBands(msi).addBands(ndwi).addBands(evi).addBands(msavi).addBands(tasseled_cap)\
    .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'SWIRratio', 'NDMI', 'NBR', 'NBR2','MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'],
            ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'SWIRratio', 'NDMI', 'NBR', 'NBR2','MSI', 'NDWI', 'EVI', 'MSAVI', 'tcB', 'tcG', 'tcW'])\
    .set('YYYYDDD', ee.Date(img.get('system:time_start')).format('YYYYDDD')) 
    return out_img



# Function to process a single tileFEAT
def process_tile(tileFEAT, cover_mask, year, samples=5, month_start=1, month_end=12, filter_double_counts=False):
    geom   = tileFEAT.GetGeometryRef()
    
    geom_json = json.loads(geom.ExportToJson())
    geom_coord = geom_json['coordinates']
    geom_coord = [c[0:2] for c in geom_coord[0]]

    geom_EE = ee.Geometry.Polygon(coords=geom_coord)
    
    # Generate a random seed
    random_seed = random.randint(0, 999999)  # You can adjust the range as needed

    training_data = cover_mask.selfMask().stratifiedSample(
        numPoints=samples,
        classBand='trainingClass',
        region=geom_EE,
        scale=30,
        seed=random_seed,
        geometries=True
    )
    LND = Seasonal_STMs_C2(year, geom_EE, month_start, month_end, filter_double_counts=filter_double_counts, reducer="all")
    vals = LND.sampleRegions(
        collection=training_data,
        properties=['trainingClass'],
        scale=30,
        tileScale=4,
        geometries=True
    ).getInfo()

    # Initialize empty lists to store x and y coordinates
    x_coordinates = []
    y_coordinates = []

    # Loop through the featureValues and extract coordinates
    for f in vals['features']:
        geometry = f['geometry']
        coordinates = geometry['coordinates']
        x_coord = coordinates[0]  # Extract the x coordinate
        y_coord = coordinates[1]  # Extract the y coordinate
        x_coordinates.append(x_coord)
        y_coordinates.append(y_coord)

    featureValues = vals['features']
    out_list = [f['properties'] for f in featureValues]

    # Add x and y coordinates to the output list
    for i, prop in enumerate(out_list):
        prop['x_coord'] = x_coordinates[i]
        prop['y_coord'] = y_coordinates[i]
        prop['year'] = year
    
    return out_list


# Code for sampling LUCAS points in GEE
def filter_LUCAS(LUCAS, cover_class = [ "artificial_land",
                                        "cropland",
                                        "woodland",
                                        "shrubland",
                                        "grassland",
                                        "bare_land",
                                        "water",
                                        "wetland"] ):
   
    # Convert class name string to LUCAS code for LC
    LC_classifications = {
    "artificial_land": "A",
    "cropland": "B",
    "woodland": "C",
    "shrubland": "D",
    "grassland": "E",
    "bare_land": "F",
    "water": "G",
    "wetland": "H"
    }

    # create selector for LC type
    LC_classifications = pd.DataFrame(LC_classifications.items(), columns=['Land Cover Type', 'Classification'])
    classification = LC_classifications[LC_classifications['Land Cover Type'] == cover_class]['Classification'].values[0]

    # Create filter conditions for lc1 and lc2 separately (LC2 commented out, due to relative unimportance). Dominant LC will be LC1 if its cover is more than 50%
    lc1_filter = ee.Filter.stringContains('lc1', classification)
    #lc2_filter = ee.Filter.stringContains('lc2', classification)

    # Filter where lc1_perc or lc2_perc are equal to "> 75%" (and don't apply this to year 2006 where the perc cover is missing)
    year_filter = ee.Filter.eq('year', 2006)
    lc1_perc_filter = ee.Filter.Or(ee.Filter.equals('lc1_perc', "> 75 %"), year_filter)
    #lc2_perc_filter = ee.Filter.Or(ee.Filter.equals('lc2_perc', "> 75 %"), year_filter)

    # Combine the two filter conditions with 'or' to filter the FeatureCollection
    filter_lc1 = ee.Filter.And(lc1_filter, lc1_perc_filter)
    #filter_lc2 = ee.Filter.And(lc2_filter, lc2_perc_filter)

    # Add a filter to exclude features where parcel_area_ha is equal to desired plot sizes 
    parcel_area_filter = ee.Filter.Or(ee.Filter.inList('parcel_area_ha', ["1 - 10 ha", "> 10 ha"]),  year_filter)
    
    # Observation quality filter. Select only in-situ points with close and accurate gps coords, or In office Photo interpritation
    filter_office_PI = ee.Filter.eq('obs_type', "In office PI")
    filter_gps_prec = ee.Filter.Or(ee.Filter.lt('gps_prec', 15), filter_office_PI)
    filter_obs_dist = ee.Filter.Or(ee.Filter.lt('obs_dist', 15), filter_office_PI)

    # Combine the filter conditions
    combined_filter = ee.Filter.And(filter_lc1, parcel_area_filter, filter_gps_prec, filter_obs_dist)

    # Filter the feature collection
    filtered_LUCAS = LUCAS.filter(combined_filter)
    
    return filtered_LUCAS

# Code for sampling LUCAS points in GEE
def filter_LUCAS_active_fallow(LUCAS, cover_class = [ "agriculture", "fallow"] ):
   
    # Convert class name string to LUCAS code for LC
    LU_classifications = {
    "agriculture": "U111",
    "fallow": "U112"
    }

    # create selector for LC type
    LU_classifications = pd.DataFrame(LU_classifications.items(), columns=['Land Use Type', 'Classification'])
    classification = LU_classifications[LU_classifications['Land Use Type'] == cover_class]['Classification'].values[0]

    # Create filter conditions for lc1 and lc2 separately (LC2 commented out, due to relative unimportance). Dominant LC will be LC1 if its cover is more than 50%
    lu1_filter = ee.Filter.stringContains('lu1', classification)
    #lc2_filter = ee.Filter.stringContains('lc2', classification)

    # Filter where lc1_perc or lc2_perc are equal to "> 75%" (and don't apply this to year 2006 where the perc cover is missing)
    year_filter = ee.Filter.eq('year', 2006)
    lu1_perc_filter = ee.Filter.Or(ee.Filter.equals('lu1_perc', "> 90 %"), year_filter)

    # Combine the two filter conditions with 'or' to filter the FeatureCollection
    filter_lc1 = ee.Filter.And(lu1_filter, lu1_perc_filter)

    # Add a filter to exclude features where parcel_area_ha is equal to desired plot sizes 
   # parcel_area_filter = ee.Filter.Or(ee.Filter.inList('parcel_area_ha', ["1 - 10 ha", "> 10 ha"]),  year_filter)
    
    # Observation quality filter. Select only in-situ points with close and accurate gps coords, or In office Photo interpritation
    filter_office_PI = ee.Filter.eq('obs_type', "In office PI")
    filter_gps_prec = ee.Filter.Or(ee.Filter.lt('gps_prec', 15), filter_office_PI)
    filter_obs_dist = ee.Filter.Or(ee.Filter.lt('obs_dist', 15), filter_office_PI)

    # Combine the filter conditions
   # combined_filter = ee.Filter.And(lu1_filter, parcel_area_filter, filter_gps_prec, filter_obs_dist)
    combined_filter = ee.Filter.And(lu1_filter, filter_gps_prec, filter_obs_dist)
    # Filter the feature collection
    filtered_LUCAS = LUCAS.filter(combined_filter)
    
    return filtered_LUCAS

# Define a function to find the closest year in lucas_years that is larger than the given year
def LUCAS_find_closest_larger_year(year, LUCAS_years):
    larger_years = [y for y in LUCAS_years if y >= year]
    if not larger_years:
        return max(LUCAS_years) 
    return min(larger_years)

def process_tile_LUCAS(tileFEAT, LUCAS, cover_mask, year, month_start=1, month_end=12, samples=5, filter_double_counts= False, LUCAS_years=[]):
    geom = tileFEAT.GetGeometryRef()
    
    # Build the EE-feature via the JSON conversion
    geom_json = json.loads(geom.ExportToJson())
    geom_coord = geom_json['coordinates']
    geom_coord = [c[0:2] for c in geom_coord[0]]
    geom_EE = ee.Geometry.Polygon(coords=geom_coord)
    
    year_lucas = None
    # Check if the provided year is in the set lucas_years
    if year not in LUCAS_years:
        # Find the closest year that is larger than the provided year
        year_lucas = LUCAS_find_closest_larger_year(year, LUCAS_years)
    else: 
        year_lucas = year
        
  #  print(year_lucas)

    # Get LUCAS data for the tile and filter it for the most appropriate year
    LUCAS_filtered = LUCAS.filterBounds(geom_EE).filter(ee.Filter.eq("year", year_lucas))
    
    # Limit the collection to the first 'samples' features
    LUCAS_filtered = LUCAS_filtered.limit(samples)
    #num_points = LUCAS_filtered.size()
    #count = ee.Number(num_points).getInfo()
    #print(count)

    # Update the year in the Seasonal_STMs_C2 function
    LND = Seasonal_STMs_C2(year, geom_EE, month_start, month_end, filter_double_counts=filter_double_counts, reducer="all")
    
        # add elevation and Bio Climactic data
    DEM           = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(geom_EE).select(["DEM"]).mosaic()
    WorldClim_BIO = ee.Image("WORLDCLIM/V1/BIO").clip(geom_EE)
    
    LND = LND.addBands(DEM).addBands(WorldClim_BIO).updateMask(cover_mask)
    
    vals = LND.sampleRegions(
        collection=LUCAS_filtered,
        properties=["id", "point_id", 'lc1_label'],
        scale=30,
        tileScale=4,
        geometries=True
    ).getInfo()

    if not vals['features']:
        return None
    else:
        # Initialize lists to store coordinates and data
        coordinates = []
        data_list = []

        for f in vals['features']:
            geometry = f['geometry']['coordinates']
            x_coord, y_coord = geometry[0], geometry[1]

            prop = f['properties']
            prop['x']    = x_coord
            prop['y']    = y_coord
            prop['year'] = year

            data_list.append(prop)

        # Create a DataFrame from the list of dictionaries
        tile_data = pd.DataFrame(data_list)

        return tile_data


