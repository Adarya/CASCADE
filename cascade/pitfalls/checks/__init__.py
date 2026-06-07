"""
CASCADE Pitfall Checks
======================

Individual check modules for each automatable pitfall class.
"""

from .comment_header import check_comment_corruption
from .covariate_leakage import check_landmark_leakage
from .collinearity import check_collinearity
from .separation import check_separation_problems
from .constant_variable import check_constant_variables
from .singular_matrix import check_singular_matrix

__all__ = [
    "check_comment_corruption",
    "check_landmark_leakage",
    "check_collinearity",
    "check_separation_problems",
    "check_constant_variables",
    "check_singular_matrix",
]
