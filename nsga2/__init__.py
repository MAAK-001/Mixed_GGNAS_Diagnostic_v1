"""Two independent Pymoo NSGA-II search branches."""

from .nsga2_seg_jacobian import run_search as run_jacobian_search
from .nsga2_spatial_swap import run_search as run_spatial_search

__all__ = ["run_spatial_search", "run_jacobian_search"]
