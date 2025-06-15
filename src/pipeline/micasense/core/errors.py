"""
MicaSense Error Module
Defines error classes for the MicaSense processing pipeline
"""

class MicaSenseError(Exception):
    """Base class for all MicaSense errors"""
    pass

class ConfigurationError(MicaSenseError):
    """Configuration related errors"""
    pass

class InputError(MicaSenseError):
    """Input data related errors"""
    pass

class ValidationError(MicaSenseError):
    """Data validation errors"""
    pass

class ProcessingError(MicaSenseError):
    """Processing pipeline errors"""
    pass

class OutputError(MicaSenseError):
    """Output file generation errors"""
    pass

class BandError(MicaSenseError):
    """Band processing errors"""
    pass

class CalibrationError(MicaSenseError):
    """Radiometric calibration errors"""
    pass

class TimeoutError(MicaSenseError):
    """Operation timeout errors"""
    pass

class GPSError(MicaSenseError):
    """GPS extraction errors"""
    pass 