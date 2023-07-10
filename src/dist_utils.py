# Library Imports
import math
import xarray as xr
import rasterio as rio
from rasterio.merge import merge
import rioxarray
import hvplot.xarray
import geoviews as gv
import pyproj
import numpy as np
import geopandas as gpd
import holoviews as hv
import folium
import panel.widgets as pnw
from shapely.geometry import shape, Point, Polygon
import datetime
from datetime import date, timedelta, datetime
from netrc import netrc
from subprocess import Popen
from getpass import getpass
import os
import tempfile
from http import cookiejar
from urllib import request
from urllib.parse import urlencode


# Functions
# The below two functions check_netrc() and make_manager() could be streamlined into one function that does both.
def check_netrc():
    # -----------------------------------AUTHENTICATION CONFIGURATION-------------------------------- #
    urs = 'urs.earthdata.nasa.gov'    # Earthdata URL to call for authentication
    prompts = ['Enter NASA Earthdata Login Username \n(or create an account at urs.earthdata.nasa.gov): ',
           'Enter NASA Earthdata Login Password: ']

    # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials
    try:
        netrcDir = os.path.expanduser("~/.netrc")
        netrc(netrcDir).authenticators(urs)[0]
        print('netrc exists and includes NASA Earthdata login credentials.')
        
    # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
    except FileNotFoundError:
        homeDir = os.path.expanduser("~")
        Popen('touch {0}.netrc | chmod og-rw {0}.netrc | echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)

    # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
    except TypeError:
        homeDir = os.path.expanduser("~")
        Popen('echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)     

    return netrc(netrcDir).authenticators(urs)

def make_manager(authenticators):
    # Create a password manager to deal with the 401 response that is returned from Earthdata Login
    password_manager = request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", authenticators[0], authenticators[2])

    # Create a cookie jar for storing cookies. This is used to store and return
    # the session cookie given to use by the data server (otherwise it will just
    # keep sending us back to Earthdata Login to authenticate). Ideally, we
    # should use a file based cookie jar to preserve cookies between runs. This
    # will make it much more efficient.
    cookie_jar = cookiejar.CookieJar()
    # Install all the handlers.
    opener = request.build_opener(
       request.HTTPBasicAuthHandler(password_manager),
       request.HTTPCookieProcessor(cookie_jar))
    request.install_opener(opener)

def stack_bands(bandpath:str, bandlist:list): 
    '''
    Returns geocube with three bands stacked into one multi-dimensional array.
            Parameters:
                    bandpath (str): Path to bands that should be stacked
                    bandlist (list): Three bands that should be stacked
            Returns:
                    bandStack (xarray Dataset): Geocube with stacked bands
                    crs (int): Coordinate Reference System corresponding to bands


            Updates: Changed load data library from xarray to rioxarray due to deprecation of xarray.open_rasterio().
            This required excluding the .scales method as well, which may cause problems, but I will wait and see.
    '''
    bandStack = []; bandS = []; bandStack_ = [];
    for i,band in enumerate(bandlist):
        if i==0:
            #bandStack_ = xr.open_rasterio(bandpath%band)
            bandStack_ = rioxarray.open_rasterio(bandpath%band)
            #crs = pyproj.CRS.to_epsg(pyproj.CRS.from_proj4(bandStack_.crs))
            crs = bandStack_.rio.crs.to_epsg()
            #bandStack_ = bandStack_ * bandStack_.scales[0]
            bandStack = bandStack_.squeeze(drop=True)
            bandStack = bandStack.to_dataset(name='z')
            bandStack.coords['band'] = i+1
            bandStack = bandStack.rename({'x':'longitude', 'y':'latitude', 'band':'band'})
            bandStack = bandStack.expand_dims(dim='band')  
        else:
            #bandS = xr.open_rasterio(bandpath%band)
            bandS = rioxarray.open_rasterio(bandpath%band)
            #bandS = bandS * bandS.scales[0]
            bandS = bandS.squeeze(drop=True)
            bandS = bandS.to_dataset(name='z')
            bandS.coords['band'] = i+1
            bandS = bandS.rename({'x':'longitude', 'y':'latitude', 'band':'band'})
            bandS = bandS.expand_dims(dim='band')
            bandStack = xr.concat([bandStack, bandS], dim='band')
    return bandStack, crs

def time_and_area_cube(dist_status, dist_date, veg_anom_max, anom_threshold, pixel_area, bounds, starting_day, ending_day, ref_date, step=3):
    '''
    Returns geocube with time and area dimensions.
            Parameters:
                    dist_status (xarray DataArray): Disturbance Status band
                    anom_max (xarray DataArray): Maximum Anomaly band
                    dist_date (xarray DataArray): Disturbance Date band
                    anom_threshold (int): Filter out pixels less than the value
                    pixel_area (float): Area of one pixel (m)
                    bounds (list): Boundary of the area of interest (pixel value)
                    starting_day (int): First observation date
                    ending_day (int): Last observation date
                    ref_date (datetime): Date of the beginning of the record
                    step (int): Increment between each day in time series

            Returns:
                    wildfire_extent (xarray Dataset): Geocube with time and area dimensions
    '''
    lats = np.array(dist_status.latitude)
    lons = np.array(dist_status.longitude)
    expanded_array1 = []
    expanded_array2 = []
    respective_areas = {}

    for i in range(starting_day, ending_day, step):
        vg = dist_status.where((veg_anom_max > anom_threshold) & (dist_date > starting_day) & (dist_date <= i))
        extent_area = compute_area(vg.data,bounds,pixel_area, ref_date)
        date = standard_date(str(i), ref_date)
        coords =  {'lat': lats, 'lon': lons, 'time': date, 'area':extent_area}
        time_and_area = xr.DataArray(vg.data, coords=coords, dims=['lat', 'lon'])
        expanded_time_and_area = xr.concat([time_and_area], 'time')
        expanded_time_and_area = expanded_time_and_area.to_dataset(name='z')
        expanded_array2.append(expanded_time_and_area)
    area_extent = xr.concat(expanded_array2[:], dim='time')
    return area_extent


def compute_area(data_,bounds, pixel_area, ref_date):
    '''
    Returns area of wildfire extent for single day.
            Parameters:
                    data (numpy array): Dist Status values (1.0-4.0)
                    bounds (list): Boundary of the area of interest (pixel value)
                    pixel_area (float): Area of one pixel (m)
            Returns:
                    fire_area (str): Wildfire extent area in kilometers squared
    '''
    data = data_[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    fire_pixel_count = len(data[np.where(data>0)])
    fire_area = fire_pixel_count * pixel_area * pow(10, -6)
    fire_area = str(math.trunc(fire_area)) + " kilometers squared"
    return fire_area

def standard_date(day, ref_date):
    '''
    Returns the inputted day number as a standard date.
            Parameters:
                    day (str): Day number that should be converted
                    ref_date (datetime): Date of the beginning of the record
            Returns:
                    res (str): Standard date corresponding to inputted day
    '''
    day.rjust(3 + len(day), '0')
    res_date = ref_date + timedelta(days=int(day))
    res = res_date.strftime("%m-%d-%Y")
    return res

def getbasemaps():
    # Add custom base maps to folium
    basemaps = {
        'Google Maps': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Maps',
            overlay = False,
            control = True,
            show = False,
        ),
        'Google Satellite': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True,
            #opacity = 0.8,
            show = False
        ),
        'Google Terrain': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Terrain',
            overlay = False,
            control = True,
            show = False,
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr = 'Google',
            name = 'Google Satellite',
            overlay = True,
            control = True,
            #opacity = 0.8,
            show = False
        ),
        'Esri Satellite': folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = True,
            control = True,
            #opacity = 0.8,
            show = False
        )
    }

    return basemaps

def handle_draw(target, action, geo_json):

    return

def merge_rasters(input_files, output_file):
    """
    Function to take a list of raster tiles, mosaic them using rasterio, and output the file.
    :param input_files: list of input raster files 
    """

    # Open the input rasters and retrieve metadata
    src_files = [rio.open(file) for file in input_files]
    meta = src_files[0].meta
    
    #mosaic the src_files
    mosaic, out_trans = merge(src_files)

    # Update the metadata
    out_meta = meta.copy()
    out_meta.update({"driver": "GTiff", 
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2], 
                    "transform": out_trans
                    }
                    )

    with rio.open(output_file, 'w', **out_meta) as dst:
        dst.write(mosaic)

    #Close the input rasters
    for src in src_files:
        src.close()
    
    return mosaic