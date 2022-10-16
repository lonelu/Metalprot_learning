"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains utility functions and error classes.
"""

class Error(Exception):
    """Base class for other exceptions"""
    pass 

class AlignmentError(Error):
    """Raised when identification of unique cores fails"""
    pass

class EncodingError(Error):
    """Raised when unrecognized amino acid is found during encoding"""
    pass

class PermutationError(Error):
    """Raised when an issue during permutation occurs"""

class ModelTypeError(Error):
    """Raised when an unrecognized value for config['model_type'] is provided"""
