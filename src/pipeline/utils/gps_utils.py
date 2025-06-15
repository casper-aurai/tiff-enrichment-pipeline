import subprocess
import re
from datetime import datetime
from typing import Optional, Dict
import logging

def parse_gps_coordinate(coord_str: str) -> float:
    if not coord_str:
        return 0.0
    coord_str = coord_str.strip()
    # Try to match DMS format: 52 deg 6' 28.48" N
    dms_pattern = re.compile(r"([\d.]+)\s*deg\s*([\d.]+)'\s*([\d.]+)\"?\s*([NSEW])", re.IGNORECASE)
    match = dms_pattern.match(coord_str)
    if match:
        deg, minutes, seconds, direction = match.groups()
        try:
            deg = float(deg)
            minutes = float(minutes)
            seconds = float(seconds)
            decimal = deg + minutes / 60 + seconds / 3600
            if direction.upper() in ['S', 'W']:
                decimal = -decimal
            return decimal
        except Exception:
            return 0.0
    # Try to match decimal with direction (e.g., 52.1234N)
    dec_pattern = re.compile(r"([\d.]+)\s*([NSEW])", re.IGNORECASE)
    match = dec_pattern.match(coord_str)
    if match:
        value, direction = match.groups()
        try:
            decimal = float(value)
            if direction.upper() in ['S', 'W']:
                decimal = -decimal
            return decimal
        except Exception:
            return 0.0
    # Try to parse as plain float
    try:
        return float(coord_str)
    except Exception:
        return 0.0

def extract_gps_info(image_path: str) -> Optional[Dict]:
    """Extract GPS info from TIFF using ExifTool plain text output. Returns dict or None."""
    cmd = [
        "exiftool",
        str(image_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logger = logging.getLogger("gps_utils")
        logger.info(f"ExifTool output for {image_path} (returncode={proc.returncode}):\n{proc.stdout}")
        if proc.returncode != 0:
            return None
        lat_str, lon_str, alt_str, ts_str = None, None, None, None
        for line in proc.stdout.splitlines():
            if 'GPS Latitude' in line and 'Ref' not in line:
                lat_str = line.split(':', 1)[1].strip()
                logger.info(f"Raw GPS Latitude line: {line}")
            elif 'GPS Longitude' in line and 'Ref' not in line:
                lon_str = line.split(':', 1)[1].strip()
                logger.info(f"Raw GPS Longitude line: {line}")
            elif 'GPS Altitude' in line and 'Ref' not in line:
                alt_str = line.split(':', 1)[1].strip()
            elif 'Date/Time Original' in line:
                ts_str = line.split(':', 1)[1].strip()
        lat = parse_gps_coordinate(lat_str) if lat_str else None
        lon = parse_gps_coordinate(lon_str) if lon_str else None
        logger.info(f"Parsed latitude: {lat} from '{lat_str}'")
        logger.info(f"Parsed longitude: {lon} from '{lon_str}'")
        try:
            # Extract numeric part from altitude string (e.g., '26.9 m Above Sea Level')
            if alt_str:
                match = re.search(r"[-+]?[0-9]*\.?[0-9]+", alt_str)
                alt = float(match.group(0)) if match else None
            else:
                alt = None
        except Exception:
            alt = None
        timestamp = None
        if ts_str:
            try:
                timestamp = datetime.strptime(ts_str, "%Y:%m:%d %H:%M:%S").isoformat()
            except Exception:
                timestamp = ts_str
        if lat is not None and lon is not None:
            return {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "timestamp": timestamp
            }
        return None
    except Exception:
        return None 