from osgeo import ogr
import ee, eemont
import pandas as pd
from osgeo import gdal, ogr, osr
import os, shutil

# ############################################## #
# #### TOOLS TO WORK WITH VECTOR DATA #### #
def CopySHPtoMem(path):
    '''
    Small tool to load a local vector file into memory for faster processing
    :param path: path (string) to the file location (required)
    :return: ogr object (vector file)
    '''
    drvMemV = ogr.GetDriverByName('Memory')
    f_open = drvMemV.CopyDataSource(ogr.Open(path),'')
    return f_open
def GEOM_CStrans(geom, epsg):
    '''
    Small function to take a geometry and covert it into a different coordinate system.
    Helpful for example when iterating over features/geometries and do something specific with it.
    Advice: pass a cloned geometry to the function (geom.Clone() )
    :param geom: geometry object (required)
    :param epsg: epsg code of destination CS (integer, required)
    :return: geometry object
    '''
    source_SR = geom.GetSpatialReference()
    source_SR.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_SR = osr.SpatialReference()
    target_SR.ImportFromEPSG(epsg)
    target_SR.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    trans = osr.CoordinateTransformation(source_SR, target_SR)
    geom.Transform(trans)
    return geom
def SaveGEOMtoFile(geom, outpath):
    drvV = ogr.GetDriverByName('ESRI Shapefile')
    outSHP = drvV.CreateDataSource(outpath)
    outLYR = outSHP.CreateLayer('outSHP', srs=geom.GetSpatialReference())
    outFEAT = ogr.Feature(outLYR.GetLayerDefn())
    outFEAT.SetGeometryDirectly(ogr.Geometry(wkt=str(geom)))
    outLYR.CreateFeature(outFEAT)

def CopySHPDisk(shape, outpath):
    drvV = ogr.GetDriverByName('ESRI Shapefile')
    outSHP = drvV.CreateDataSource(outpath)
    lyr = shape.GetLayer()
    sett90LYR = outSHP.CopyLayer(lyr, 'lyr')
    del lyr, shape, sett90LYR, outSHP
    
# ################################################################ #
# #### TOOLS TO WORK IN GEE #### #
    
def GetFullLandsatCollection_withIndices(roi, year):
    l4 = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .spectralIndices(['NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI']) \
        .select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'],
        ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'])

    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .spectralIndices(['NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI']) \
        .select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'],
        ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'])

    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .spectralIndices(['NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI']) \
        .select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'],
        ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'])

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .spectralIndices(['NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI']) \
        .select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'],
        ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'])

    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .spectralIndices(['NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI']) \
        .select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'],
        ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'EVI2', 'SAVI2', 'NDSI', 'NDBaI', 'kNDVI'])

    col = ee.ImageCollection(l4.merge(l5).merge(l7).merge(l8).merge(l9))
    return col

def GetFullLandsatCollection(roi, year):
    l4 = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi) \
        .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

    col = ee.ImageCollection(l4.merge(l5).merge(l7).merge(l8).merge(l9))
    return col

def GetL8_9LandsatCollection(roi, year):
    if year < 2013:
        print("Landsat 8 started in 2013, please insert a year from 2013 onwards")
    else:
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi) \
            .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
            .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi) \
            .filter(ee.Filter.date(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))).preprocess() \
            .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'])

        col = ee.ImageCollection(l8.merge(l9))
        return col

def Write_json_to_csv(json, path):
    import pandas as pd
    flag = 0
    featVals = json['features']
    for f in featVals:
        prop = f['properties']
        if flag == 0:
            # Create a pandas data frame with the first element of the values
            out_pd = pd.DataFrame(prop, index=[0])
            flag = 1
        else:
            out_pd = pd.concat([out_pd, pd.DataFrame(prop, index=[0])])
    # Write the pandas dataframe 
    out_pd.to_csv(path, index=False)

def lyrTOfc(in_lyr):
    '''
    Function, that converts a vector-layer into a feature collection, that can
    be visualized (or used otherwise) in GEE.
    PARAMS:
    ========
    in_lyr (required): layer, return object from ogr.Open()

    RETURNS:
    ========
    ee.FeatureCollection()
    '''
    # Create an emtpy feature collection
    fc = ee.FeatureCollection([])
    # Create a rule to transform the CS into EPSG:4326
    fromSR = lyr.GetSpatialRef()
    toSR = osr.SpatialReference()
    toSR.ImportFromEPSG(4326)
    cT = osr.CoordinateTransformation(fromSR, toSR)
    # Get the feature fields and geometries and add them to the feature collection
    for feat in in_lyr:
        geom = feat.GetGeometryRef()
        geom_cl = geom.Clone()
        geom_cl.Transform(cT)
        geojson = json.loads(geom_cl.ExportToJson())
        properties = {}
        for i in range(feat.GetFieldCount()):
            field_name = feat.GetFieldDefnRef(i).GetName()
            field_value = feat.GetField(i)
            properties[field_name] = field_value
        ee_feat = ee.Feature(ee.Geometry(geojson), properties)
        # Remove the 'id' property from the feature
        ee_feat = ee_feat.set('system:index', None)
        fc = fc.merge(ee.FeatureCollection(ee_feat))
    return fc

    
def fcTOshp(fc_geojson, outName):
    '''
    Function that takes the output from the getInfo of a ee.FeatureCollection() and
    converts it into a local shapefile. Code was generated through ChatGPT
    !!!! Works so far only for homogenous shapefiles (i.e., Point OR Polygon OR LineString!!!
    
    Arguments:
    ----------
    fc_geojson (required): output from getInfo() calling to a featureCollection
    outName (string, required): Name of the desired output-file
    
    '''
    # Check which geomtry type we have
    firstFeat = fc_geojson['features'][0]
    firstGeom = firstFeat['geometry']['type']   
    # Add the spatial reference
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    # Open the output shapefile using OGR
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapefile = driver.CreateDataSource(outName)
    if firstGeom == 'Polygon':
        layer = shapefile.CreateLayer(outName.split('.')[0], sr, geom_type=ogr.wkbPolygon)
    if firstGeom == 'Point':
        layer = shapefile.CreateLayer(outName.split('.')[0], sr, geom_type=ogr.wkbPoint)
    if firstGeom == 'LineString':
        layer = shapefile.CreateLayer(outName.split('.')[0], sr, geom_type=ogr.wkbLineString)
    # Get the feature fields and add them to the output shapefile attribute table
    for field in fc_geojson['features'][0]['properties'].keys():
        layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))
    # Add the feature geometries and attributes to the output shapefile
    for feature in fc_geojson['features']:
        geom = ogr.CreateGeometryFromJson(str(feature['geometry']))
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetGeometry(geom)
        for field in feature['properties'].keys():
            feat.SetField(field, feature['properties'][field])
        layer.CreateFeature(feat)
        feat = None
    # Close the output shapefile
    shapefile = None
    layer = None

def fcTOshpMEM(fc_geojson):
    '''
    Same function as above, but here we get a shapefile in memory in return
    
    Arguments:
    ----------
    fc_geojson (required): output from getInfo() calling to a featureCollection
    
    '''
    # Check which geomtry type we have
    firstFeat = fc_geojson['features'][0]
    firstGeom = firstFeat['geometry']['type']   
    # Add the spatial reference
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    # Open the output shapefile using OGR
    driver = ogr.GetDriverByName('Memory')
    shapefile = driver.CreateDataSource('')
    if firstGeom == 'Polygon':
        layer = shapefile.CreateLayer('', geom_type=ogr.wkbPolygon)
    if firstGeom == 'Point':
        layer = shapefile.CreateLayer('', sr, geom_type=ogr.wkbPoint)
    if firstGeom == 'LineString':
        layer = shapefile.CreateLayer('', sr, geom_type=ogr.wkbLineString)
    # Get the feature fields and add them to the output shapefile attribute table
    for field in fc_geojson['features'][0]['properties'].keys():
        layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))
    # Add the feature geometries and attributes to the output shapefile
    for feature in fc_geojson['features']:
        geom = ogr.CreateGeometryFromJson(str(feature['geometry']))
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetGeometry(geom)
        for field in feature['properties'].keys():
            feat.SetField(field, feature['properties'][field])
        layer.CreateFeature(feat)
        feat = None
    # Close the output shapefile
    return shapefile
    
# ############################################## #
# #### SMALL TOOLS TO WORK WITH RASTER DATA #### #

def GetRasterProperties(raster):
    '''
    Function, that returns a dictionary with all necessary information of a raster file
    raster: gdal Object (required)
    returns: dictionary
    '''
    pr = raster.GetProjection() # Projection
    gt = raster.GetGeoTransform() # Afine transformation
    cols = raster.RasterXSize # Number of Columns
    rows = raster.RasterYSize # Nubmber of Rows
    nbands = raster.RasterCount # Number of bands

    band = raster.GetRasterBand(1)
    dType = band.DataType # data type

    return {'pr':pr, 'gt':gt, 'cls': cols, 'rws': rows, 'nbands': nbands, 'dataType':dType}

def BuildVRT(folder, outfile, bandList="all"):
    '''
    :param folder: string (required). Folder where to search for the tiles
    :param outfile: string (required). Name of the output vrt. Has to end with .vrt
    :param bandList: list (optional). List of the bands to build the vrt with
    :return: Opened VRT (gdal object).
    '''
    if bandList == "all":
        fileList = GetFilesInFolderWithEnding(folder, ".tif", fullPath=True)
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
        vrt = gdal.BuildVRT(outfile, fileList, options=vrt_options)
        vrt = None
        vrt = gdal.Open(outfile)
    else:
        fileList = GetFilesInFolderWithEnding(folder, ".tif", fullPath=True)
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False, bandList=bandList)
        vrt = gdal.BuildVRT(outfile, fileList, options=vrt_options)
        vrt = None
        vrt = gdal.Open(outfile)
    return vrt

def SaveRASTERtoFile(memRas, outpath):
    endList = [[".tif", 'GTiff'], [".bsq", "ENVI"], [".img", "HFA"]]
    for end in endList:
        if outpath.endswith(end[0]):
            ending = end[1]
    drvR = gdal.GetDriverByName(ending)
    drvR.CreateCopy(outpath, memRas)
    memRas = None
    outpath = None
    # To add --> find the highest rastervalue and subsequently find optimal format (bit, byte, float)

def GetDataTypeHexaDec(gdalDtype):
    '''
    Tool for identifying the data type of a rasterband in form of a hexadecimal code. Is needed for many scripts
    that I wrote to do point intersections
    :param gdalDtype: as integer (1-7) or from rb.DataType (required)
    :return: data type as hexadecimal code
    '''
    dTypes = [[1, 'b'], [2, 'H'], [3, 'h'], [4, 'I'], [5, 'i'], [6, 'f'], [7, 'd']]
    for dT in dTypes:
        if dT[0] == gdalDtype:
            band_dType = dT[1]
    return band_dType

def Geom_Raster_to_np(geom, raster, band, pxSize_m):
    '''
    Function that takes a geometry from a polygon shapefile and a rasterfile, and returns both features as 2d-arryas
    in the size of the geom --> can be later used for masking.
    Function does a coordinate transformation implicitely!

    PARAMETERS
    -----------
    geom : geom object (required)
        geometry of the feature
    raster: gdal object (required)
        raster as a gdal-object (through gdal.Open())
    band: integer (required)
        the band value to take from the raster

    RETURNS
    -------
    Two numpy-arrays
    (1) np-array of the geometry as binary feature --> values inside the geometry have value '1', values outside '0'
    (2) np-array of the raster in the same size (i.e., as a subset of the raster) of the geometry

    '''
# Create a shp/lyr in memory to rasterize the geometry into it
    # Get the geometry, and rasterizee it
    lyr_sr = geom.GetSpatialReference()
    geom_shp = ogr.GetDriverByName('memory').CreateDataSource('')
    geom_lyr = geom_shp.CreateLayer('geom', geom.GetSpatialReference(), geom_type=ogr.wkbMultiPolygon)
    geom_lyr_defn = geom_lyr.GetLayerDefn()
    geom_feat = ogr.Feature(geom_lyr_defn)
    geom_feat.SetGeometry(geom)
    geom_lyr.CreateFeature(geom_feat)
    x_min, x_max, y_min, y_max = geom_lyr.GetExtent()
    x_res = int((x_max - x_min) / pxSize_m)
    y_res = int((y_max - y_min) / pxSize_m)
    geom_ras = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
    geom_ras.SetProjection(lyr_sr.ExportToWkt())
    geom_ras.SetGeoTransform((x_min, pxSize_m, 0, y_max, 0, -pxSize_m))
    gdal.RasterizeLayer(geom_ras, [1], geom_lyr, burn_values=[1])
    geom_np = geom_ras.GetRasterBand(1).ReadAsArray()
    #CopyMEMtoDisk(geom_ras, workfolder + "geom.tif")
# Now re-project the rasterfile into the same extent, then open the array
    raster_sub = gdal.GetDriverByName('MEM').Create('', geom_ras.RasterXSize, geom_ras.RasterYSize, raster.RasterCount, gdal.GDT_Byte)
    raster_sub.SetGeoTransform(geom_ras.GetGeoTransform())
    raster_sub.SetProjection(geom_ras.GetProjection())
    gdal.ReprojectImage(raster, raster_sub, raster.GetProjection(), geom_ras.GetProjection(), gdal.GRA_NearestNeighbour)
    raster_np = np.array(raster_sub.GetRasterBand(band).ReadAsArray())
    #CopyMEMtoDisk(raster_sub, workfolder + "raster.tif")
    #exit(0)
    return geom_np, raster_np

def OpenRasterToMemory(path):
    drvMemR = gdal.GetDriverByName('MEM')
    ds = gdal.Open(path)
    dsMem = drvMemR.CreateCopy('', ds)
    return dsMem

def BuildPyramids(path, levels=None):
    # levels is a vector that is multiples of 2 --> e.g., 2, 4, 6, 8, ..., 64
    if levels == None:
        command = "gdaladdo.exe -r nearest -ro " + path + " 2 4 8 16 32"
        os.system(command)
    else:
        command = "gdaladdo.exe -r nearest -ro " + path + " "
        for l in levels:
            command = command + l + " "
        os.system(command)

def ClipRasterBySHP(SHP, raster, mask=False):

    drvMemR = gdal.GetDriverByName('MEM')
    drvMemV = ogr.GetDriverByName('Memory')
## DO SOME PRE-THINGS
# Check if "SHP" is a string/path or an object. If string, then copy to memory
    if isinstance(SHP, str):
        SHP = drvMemV.CopyDataSource(ogr.Open(SHP), '')
    else:
        SHP = SHP
# Check if "raster" is a string/path or an object. If string, then copy to memory
    if isinstance(raster, str):
        raster = gdal.Open(raster)
    else:
        raster = raster
## DO THE CLIPPING
# Get the geometry-infos, and raster infos
    lyr = SHP.GetLayer()
    lyr_pr = lyr.GetSpatialRef()
    lyr_pr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ras_pr = raster.GetProjection()
    ras_gt = raster.GetGeoTransform()
    pixelSize = ras_gt[1]
    xOrigin = ras_gt[0]
    yOrigin = ras_gt[3]
    rb = raster.GetRasterBand(1)
    dType = rb.DataType
    NoDataValue = rb.GetNoDataValue()
# Get the extent of the SHP-file, apply coordinate transformation and convert into raster-coordinates
    #https://gis.stackexchange.com/questions/65840/subsetting-geotiff-with-python
    target_SR = osr.SpatialReference()
    target_SR.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_SR.ImportFromWkt(ras_pr)
    transform = osr.CoordinateTransformation(lyr_pr, target_SR)
    ext = lyr.GetExtent() #x_min, x_max, y_min, y_max
# Find largest extent after the coordinate transformation
    ulx, uly, ulz = transform.TransformPoint(ext[0], ext[3])
    llx, lly, llz = transform.TransformPoint(ext[0], ext[2])
    urx, ury, ulz = transform.TransformPoint(ext[1], ext[3])
    lrx, lry, llz = transform.TransformPoint(ext[1], ext[2])
    minX = min(ulx, llx, urx, lrx)
    maxY = max(uly, ury, lly, lry)
    maxX = max(urx, lrx, ulx, llx)
    minY = min(uly, lly, ury, lry)
# Calculate the new number of columns and rows
    i1 = int((minX - xOrigin) / pixelSize)
    j1 = int((yOrigin - maxY) / pixelSize)
    i2 = int((maxX - xOrigin) / pixelSize)
    j2 = int((yOrigin - minY) / pixelSize)
    colsNew = i2-i1
    rowsNew = j2-j1
    newX = xOrigin + i1*pixelSize
    newY = yOrigin - j1*pixelSize
# Read the raster into an array based on the raster coordinates, then create output-file in memory
    if raster.RasterCount == 1:
        array = raster.GetRasterBand(1).ReadAsArray(i1, j1, colsNew, rowsNew)
    if raster.RasterCount > 1:
        array = np.zeros((rowsNew, colsNew, raster.RasterCount))
        for bandCount, arrCount in enumerate(range(raster.RasterCount), start=1):
            array[:, :, arrCount] = raster.GetRasterBand(bandCount).ReadAsArray(i1, j1, colsNew, rowsNew)

# If mask = TRUE, then additionally mask the areas outside the polygon
    if mask == True:
# Re-project layer into CS of the raster
        # Create TMP shapefile in memory
        feat = lyr.GetNextFeature()
        geom = feat.GetGeometryRef()
        geomType = geom.GetGeometryType()
        lyr.ResetReading()
        tmpSHP = drvMemV.CreateDataSource('')
        tmpLYR = tmpSHP.CreateLayer('tmpSHP', target_SR, geom_type=geomType)
        tmpLYRDefn = tmpLYR.GetLayerDefn()
        # Now move every geometry over to the tempLYR
        inFeat = lyr.GetNextFeature()
        while inFeat:
           inGeom = inFeat.GetGeometryRef()
           inGeom.Transform(transform)
           tmpFeat = ogr.Feature(tmpLYRDefn)
           tmpFeat.SetGeometry(inGeom)
           tmpLYR.CreateFeature(tmpFeat)
           tmpFeat = None
           inFeat = lyr.GetNextFeature()
# Create a array mask from the temporary shapefile
        shpRas = drvMemR.Create('', colsNew, rowsNew, gdal.GDT_Byte)
        shpRas.SetProjection(ras_pr)
        shpRas.SetGeoTransform((newX, pixelSize, ras_gt[2], newY, ras_gt[4], -pixelSize))
        shpRasBand = shpRas.GetRasterBand(1)
        shpRasBand.SetNoDataValue(0)
        gdal.RasterizeLayer(shpRas, [1], tmpLYR, burn_values=[1])
        # root_folder = "E:/Baumann/_ANALYSES/AnnualLandCoverChange_CHACO/"
        # classRun = 6
        # out_root = root_folder + "04_Map_Products/Run" + str("{:02d}".format(classRun)) + "/"
        # outname = out_root + "Run" + str("{:02d}".format(classRun)) + "maskkk.tif"
        # bt.baumiRT.CopyMEMtoDisk(shpRas, outname)
        # exit(0)
        shpArray = shpRasBand.ReadAsArray()
    ### Test: in case of multiband do this per band
        if array.ndim == 2:
            array = np.where((shpArray == 0), 0, array)
        if array.ndim > 2:
            import copy
            for index in range(array.shape[2]):
                array_index = array[:, :, index]
                array_index = np.where((shpArray == 0), 0, array_index)
                array[:, :, index] = copy.deepcopy(array_index)
# Mask the array       
    else:
        array = array
        
# Write to disc --> again, check if more than one band
    if array.ndim == 2:
        outRas = drvMemR.Create('', colsNew, rowsNew, 1, dType)
        outRas.SetProjection(ras_pr)
        outRas.SetGeoTransform((newX, pixelSize, ras_gt[2], newY, ras_gt[4], -pixelSize))
        # write the values into it
        outRas.GetRasterBand(1).WriteArray(array)
        #outRas.GetRasterBand(1).SetNoDataValue(NoDataValue)
    if array.ndim > 2:
        outRas = drvMemR.Create('', colsNew, rowsNew, array.shape[2], dType)
        outRas.SetProjection(ras_pr)
        outRas.SetGeoTransform((newX, pixelSize, ras_gt[2], newY, ras_gt[4], -pixelSize))
        # write the values into it
        for bandCount, arrCount in enumerate(range(array.shape[2]), start=1):
            outRas.GetRasterBand(bandCount).WriteArray(array[:, :, arrCount])
        
    return outRas


# ######################################################## #
# #### SMALL TOOLS TO WORK WITH FILES AN FILE SYSTEMS #### #

def CreateFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        print("Folder already exists")
    return path

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

def GetFilesInFolderWithEnding(folder, ext, fullPath):
    ''' Function that returns all filenames in a folder that match the extension

        Parameters
    ----------
    folder : string (required)
        path, through which we are searching
    ext : string (required)
        extension of the files that we want to search for
    fullPath : bool (required)
        option to return just the file names or the entire file path
        if True, then all matched files are concatenated with 'folder'

    Returns
    -------
    outlist : list of strings
        CAUTION: if len(outlist) == 1, then outlist is a variable, will be returned with a print-statement
    '''

    outlist = []
    input_list = os.listdir(folder)
    if fullPath == True:
        for file in input_list:
            if file.endswith(ext):
    # Check if the variable folder ends with a '/', otherwise manually add to get correct path
                if folder.endswith("/"):
                    filepath = folder + file
                else:
                    filepath = folder + "/" + file
                outlist.append(filepath)
    if fullPath == False or fullPath == None:
        for file in input_list:
            if file.endswith(ext):
                outlist.append(file)
    if len(outlist) == 1:
        print("Found only one file matching the extension. Returning a variable instead of a list")
        outlist = outlist[0]
    if len(outlist) == 0:
        print("Could not find any file matching the extension. Return-value is None")
    return outlist
