"""Data for generating test images and signals."""

import sys
from cycler import cycler

from ._dim1 import sin_trapezoid, piecewise_constant_1d_test_problem, mixed_test_problem, comp_emp_bayes_t1d_test_problem
from ._dim2 import read_image, ge, cameraman, grandcanyon, shepplogan, seaice, fingerprint
from ._dim2 import cortex, satellite, mri, graecolatinsquare, sar1, sar2, meme


__all__ = [
    "read_image",
    "ge",
    "cameraman",
    "grandcanyon",
    "shepplogan",
    "seaice",
    "fingerprint",
    "cortex",
    "satellite",
    "mri",
    "sin_trapezoid",
    "graecolatinsquare",
    "sar1",
    "sar2",
    "piecewise_constant_1d_test_problem",
    "meme",
    "mixed_test_problem",
    "comp_emp_bayes_t1d_test_problem",
]


# Also define dartmouth colors
dartmouth_colors = {
    "dartmouth_green": "#00693e",
    "forest_green": "#12312b",
    "rich_forest_green": "#0D1E1C",
    "snow_white": "#ffffff",
    "midnight_black": "#000000",
    "web_gray_1": "#f7f7f7",
    "web_gray_2": "#e2e2e2",
    "web_gray_3": "#707070",
    "granite_gray": "#424141",
    "autumn_brown": "#643c20",
    "bonfire_red": "#9d162e",
    "tuck_orange": "#e32d1c",
    "summer_yellow": "#f5dc69",
    "spring_green": "#c4dd88",
    "river_navy": "#003c73",
    "river_blue": "#267aba",
    "web_violet": "#8a6996",
    "bonfire_orange": "#ffa00f",
}


dartmouth_color_cycler = cycler( color = [
        dartmouth_colors["river_blue"],
        dartmouth_colors["bonfire_orange"],
        dartmouth_colors["spring_green"],
        dartmouth_colors["bonfire_red"],
        dartmouth_colors["web_violet"],
        dartmouth_colors["autumn_brown"],
        dartmouth_colors["web_gray_2"],
        dartmouth_colors["summer_yellow"],
        #dartmouth_colors["spring_green"],
        #dartmouth_colors["river_blue"],
        
    ]
)




for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__









