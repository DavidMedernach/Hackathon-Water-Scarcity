"""Load raw data for the hydrological model."""

import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr


def load_hydro_data(area, dir):
    """Load hydro area data for a given Level of the hydrographic network.

    Args:
      area (str): France or Brazil
      dataset_dir (str): Base directory containing the data

    Returns:
      dict: A dictionary containing 4 hydrographic levels
    """
    path = f"{dir}{area}/static_data/hydro_areas/"

    if area == "france":
        region_file = f"{path}RegionHydro.json"
        sector_file = f"{path}SecteurHydro.json"
        sub_sector_file = f"{path}SousSecteurHydro.json"
        zone_file = f"{path}ZoneHydro.json"
    elif area == "brazil":
        region_file = f"{path}geoft_bho_ach_otto_nivel_02.gpkg"
        sector_file = f"{path}geoft_bho_ach_otto_nivel_03.gpkg"
        sub_sector_file = f"{path}geoft_bho_ach_otto_nivel_04.gpkg"
        zone_file = f"{path}geoft_bho_ach_otto_nivel_05.gpkg"
    else:
        raise ValueError(f"Area '{area}' not supported.")

    hydro_data = {
        "region": gpd.read_file(region_file),
        "sector": gpd.read_file(sector_file),
        "sub_sector": gpd.read_file(sub_sector_file),
        "zone": gpd.read_file(zone_file),
    }

    return hydro_data


def read_soil_data(area, dir):
    """Read soil data for a given area.

    Args:
      area (str): France or Brazil
      dir (str): Base directory containing the data

    Returns:
      xarray.Dataset: The soil data
    """
    path = f"{dir}{area}/static_data/soil"
    soil_file = f"{path}/{area}_soil_data.nc"
    return xr.open_dataset(soil_file)


def read_altitude_data(area, dir):
    """Read altitude data (DEM) for a given area.

    Args:
      area (str): France or Brazil
      dir (str): Base directory containing the data

    Returns:
      xarray.DataArray: The altitude data
    """
    path = f"{dir}{area}/static_data/altitude_DEM/output_SRTMGL1.tif"
    dem = rioxarray.open_rasterio(path, masked=True).squeeze()
    return dem


def load_meteo_data(area, meteo_type, dir):
    """Load meteo data for a given area and type.

    Args:
      area (str) : France or Brazil
      meteo_type (str) : Type of data (the key used in `datasets`)
      dir (str) : Base directory containing the data

    Returns :
      dict : A dictionary containing the xArrays for:
          - 'precipitations'
          - 'temperatures'
          - 'soil_moisture'
          - 'evaporation'
    """
    path = f"{dir}{area}/{meteo_type}/meteo"
    precipitations = xr.open_dataset(f"{path}/total_precipitation.nc")
    temperatures = xr.open_dataset(f"{path}/2m_temperature.nc")
    soil_moisture = xr.open_dataset(f"{path}/volumetric_soil_water_layer_1.nc")
    evaporation = xr.open_dataset(f"{path}/evaporation.nc")

    return {
        "precipitations": precipitations,
        "temperatures": temperatures,
        "soil_moisture": soil_moisture,
        "evaporation": evaporation,
    }


def load_station_info(area, meteo_type, dir):
    """Load station info for a given area and type.

    Args:
      area (str) : France or Brazil
      meteo_type (str) : Type of data (the key used in `datasets`)
      dir (str) : Base directory containing the data

    Returns :
      pd.DataFrame : DataFrame of stations with renamed columns
        and updated altitude.
    """
    path = f"{dir}{area}/{meteo_type}/waterflow/station_info.csv"
    df = pd.read_csv(path, sep=",")
    df = df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "catchment_area (km2)": "catchment",
            "altitude (m)": "altitude",
            "Catchment area (km²)": "catchment",
            "Altitude (m ASL)": "altitude",
        }
    )
    return df


def load_water_flows(area, meteo_type, dir):
    """Load water flows for a given area and type.

    Args:
      area (str) : France or Brazil
      meteo_type (str) : Type of data (the key used in `datasets`)
      dir (str) : Base directory containing the data

    Returns :
      pd.DataFrame : DataFrame of water flows with the 'ObsDate'
        column correctly formatted.
    """
    path = f"{dir}{area}/{meteo_type}/waterflow/waterflow_data.csv"

    wf = pd.read_csv(path, sep=",")
    wf = wf.rename(columns={"date": "ObsDate"})
    wf["ObsDate"] = pd.to_datetime(wf["ObsDate"]).dt.tz_localize(None).dt.floor("d")
    return wf
