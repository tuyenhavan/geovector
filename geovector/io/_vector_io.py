import os
import pathlib
from typing import Optional
from urllib.parse import urlparse

import geopandas as gpd
import pyogrio


class VectorReader:

    SUPPORTED_EXTENSION = {".shp", ".geojson", ".gpkg", ".kml", ".sqlite", ".tab"}

    def __init__(self, path: str) -> None:
        """Initialize the VectorReader instance

        Args:
            path (str): A file path.

        Raises:
            FileNotFoundError: File not found error.
            ValueError: Unsupported file.
        """
        self.path = path
        if isinstance(path, str):
            self.ext = pathlib.Path(path).suffix
        try:
            if not os.path.exists(self.path):
                raise FileNotFoundError("No files found!")
            if self.ext not in self.SUPPORTED_EXTENSION:
                raise ValueError("Unsupported file!")
        except:
            if not self.is_link(self.path):
                raise ValueError("Unsupported file. Please recheck your link.")

    def get_data(self) -> Optional[gpd.GeoDataFrame]:
        """Read the actual data.

        Returns:
            gpd.DataFrame: A geopandas dataframe.
        """
        try:
            if self.ext in self.SUPPORTED_EXTENSION and self.ext != ".tab":
                data = gpd.read_file(self.path)
            elif self.is_link(self.path):
                data = gpd.read_file(self.path)
            else:
                data = pyogrio.read_dataframe(self.path)
            return data
        except:
            print("Can't read this file!")
            return None

    def reproject(self, crs: Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
        """Reproject a vector file to new coordinate reference system.

        Args:
            crs (Optional[str], optional): A crs of interest. Defaults to None.

        Returns:
            Optional[gpd.GeoDataFrame]: A geodataframe with a new coordinate reference system.
        """
        if crs is not None:
            if isinstance(crs, str):
                data = self.get_data().to_crs(crs)
                return data
            else:
                print(
                    "Invalid CRS format. Please provide correct CRS (e.g., 'epsg:4326')"
                )
        return None

    @staticmethod
    def is_link(link_path: str) -> bool:
        """Check if a given file path is a valid web url.

        Args:
            link_path (str): A specified file path or url to check.

        Returns:
            bool: True if the specified file is a web url and False otherwise.
        """
        link = urlparse(link_path)
        if link.scheme in ["http", "https"]:
            return True
        else:
            return False
