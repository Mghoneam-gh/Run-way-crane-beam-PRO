"""
CRANE RUNWAY BEAM DESIGN PRO V6.0 - PROFESSIONAL EDITION
=========================================================
Per AISC 360-16 (ASD), AISC Design Guide 7, CMAA 70/74

Features:
- Complete moving load analysis with all steps displayed
- Hot-rolled sections (IPE, HEA, HEB, UB, UC) 
- Built-up plate girder design
- Cap channels (UPN, PFC)
- Full AISC 360-16 design checks
- Stiffener design (transverse, bearing, longitudinal)
- Optional fatigue analysis
- Academic PDF report generation
- Run Design button for controlled execution

TO RUN:
    pip install streamlit pandas numpy plotly reportlab
    streamlit run CraneRunwayProV6.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import io

# PDF Report Generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
        PageBreak, Image, KeepTogether, HRFlowable
    )
    from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False



# ============================================================================
# MATERIAL CONSTANTS
# ============================================================================
E_STEEL = 200000    # MPa - Modulus of elasticity
G_STEEL = 77200     # MPa - Shear modulus  
GRAVITY = 9.81      # m/s²
RHO_STEEL = 7850    # kg/m³ - Steel density

# ============================================================================
# SAFETY FACTORS (ASD)
# ============================================================================
OMEGA_FLEX = 1.67       # Flexure
OMEGA_SHEAR = 1.50      # Shear (unstiffened)
OMEGA_SHEAR_TFA = 1.67  # Shear with tension field action
OMEGA_WLY = 1.50        # Web local yielding
OMEGA_WCR = 2.00        # Web crippling
OMEGA_COMP = 1.67       # Compression (stiffeners)
OMEGA_BEARING = 2.00    # Bearing

# ============================================================================
# DESIGN LIMITS
# ============================================================================
MAX_UTIL_RATIO = 1.0    # Maximum utilization ratio
MIN_STIFF_THICKNESS = 6 # mm - Minimum stiffener thickness

# ============================================================================
# WELD DESIGN DATA (AISC 360-16 & AWS D1.1)
# ============================================================================
# Electrode classifications per AWS D1.1
WELD_ELECTRODES = {
    'E60': {'FEXX': 415, 'desc': 'E60XX - 60 ksi electrode'},
    'E70': {'FEXX': 480, 'desc': 'E70XX - 70 ksi electrode (most common)'},
    'E80': {'FEXX': 550, 'desc': 'E80XX - 80 ksi electrode'},
    'E90': {'FEXX': 620, 'desc': 'E90XX - 90 ksi electrode'},
    'E100': {'FEXX': 690, 'desc': 'E100XX - 100 ksi electrode'},
    'E110': {'FEXX': 760, 'desc': 'E110XX - 110 ksi electrode'},
}

# Weld types
WELD_TYPES = {
    'fillet': {
        'name': 'Fillet Weld',
        'desc': 'Most economical for built-up sections',
        'omega': 2.00,  # ASD safety factor for fillet welds
        'phi': 0.75,    # LRFD resistance factor
    },
    'CJP': {
        'name': 'Complete Joint Penetration (CJP)',
        'desc': 'Full strength groove weld',
        'omega': 1.67,  # Tension/compression: same as base metal
        'phi': 0.90,
    },
    'PJP': {
        'name': 'Partial Joint Penetration (PJP)',
        'desc': 'Partial depth groove weld',
        'omega': 2.00,
        'phi': 0.75,
    },
}

# Minimum fillet weld sizes per AISC Table J2.4 (based on thicker part joined)
MIN_FILLET_WELD_SIZE = {
    6.35: 3,    # t ≤ 1/4" (6.35mm) -> 3mm (1/8")
    12.7: 5,    # 1/4" < t ≤ 1/2" (12.7mm) -> 5mm (3/16")
    19.05: 6,   # 1/2" < t ≤ 3/4" (19.05mm) -> 6mm (1/4")
    38.1: 8,    # 3/4" < t ≤ 1-1/2" (38.1mm) -> 8mm (5/16")
    57.15: 10,  # 1-1/2" < t ≤ 2-1/4" (57.15mm) -> 10mm (3/8")
    152.4: 13,  # t > 2-1/4" (57.15mm) -> 13mm (1/2")
}

# Maximum fillet weld sizes per AISC J2.2b
# Along edges of material less than 1/4" thick: weld size = material thickness
# Along edges of material 1/4" or more thick: weld size = material thickness - 1/16" (1.6mm)


# ============================================================================
# STEEL GRADES
# ============================================================================
STEEL_GRADES = {
    'A36': {'Fy': 250, 'Fu': 400, 'desc': 'ASTM A36 Carbon Steel'},
    'A572_Gr42': {'Fy': 290, 'Fu': 415, 'desc': 'ASTM A572 Grade 42 HSLA'},
    'A572_Gr50': {'Fy': 345, 'Fu': 450, 'desc': 'ASTM A572 Grade 50 HSLA'},
    'A992': {'Fy': 345, 'Fu': 450, 'desc': 'ASTM A992 Structural Steel'},
    'A913_Gr50': {'Fy': 345, 'Fu': 450, 'desc': 'ASTM A913 Grade 50 QST'},
    'A913_Gr65': {'Fy': 450, 'Fu': 550, 'desc': 'ASTM A913 Grade 65 QST'},
    'S235': {'Fy': 235, 'Fu': 360, 'desc': 'EN 10025 S235'},
    'S275': {'Fy': 275, 'Fu': 430, 'desc': 'EN 10025 S275'},
    'S355': {'Fy': 355, 'Fu': 510, 'desc': 'EN 10025 S355'},
    'S460': {'Fy': 460, 'Fu': 540, 'desc': 'EN 10025 S460'},
}

# ============================================================================
# CRANE CLASSES (CMAA 70/74)
# ============================================================================
CRANE_CLASSES = {
    'A': {
        'name': 'Standby/Infrequent',
        'cycles': '20K-100K',
        'max_cycles': 100000,
        'defl_limit': 600,
        'desc': 'Infrequent use, precise handling - powerhouses, transformer rooms'
    },
    'B': {
        'name': 'Light',
        'cycles': '100K-500K',
        'max_cycles': 500000,
        'defl_limit': 600,
        'desc': 'Light service - repair shops, light assembly, service buildings'
    },
    'C': {
        'name': 'Moderate',
        'cycles': '500K-2M',
        'max_cycles': 2000000,
        'defl_limit': 600,
        'desc': 'Moderate service - machine shops, paper mills, light warehouses'
    },
    'D': {
        'name': 'Heavy',
        'cycles': '2M-10M',
        'max_cycles': 10000000,
        'defl_limit': 800,
        'desc': 'Heavy service - heavy machine shops, foundries, fabrication'
    },
    'E': {
        'name': 'Severe',
        'cycles': '10M-20M',
        'max_cycles': 20000000,
        'defl_limit': 1000,
        'desc': 'Severe service - scrap yards, cement plants, lumber mills'
    },
    'F': {
        'name': 'Continuous Severe',
        'cycles': '>20M',
        'max_cycles': 50000000,
        'defl_limit': 1000,
        'desc': 'Continuous severe service - steel mills, container handling'
    },
}

# ============================================================================
# FATIGUE CATEGORIES (AISC 360-16 Appendix 3)
# ============================================================================
FATIGUE_CATEGORIES = {
    'A': {
        'Cf': 250e8,
        'FTH': 165,
        'desc': 'Base metal with rolled or cleaned surfaces'
    },
    'B': {
        'Cf': 120e8,
        'FTH': 110,
        'desc': 'Base metal at welded connections - continuous welds'
    },
    'B\'': {
        'Cf': 61e8,
        'FTH': 83,
        'desc': 'Base metal at groove weld transitions'
    },
    'C': {
        'Cf': 44e8,
        'FTH': 69,
        'desc': 'Welded stiffeners and attachments < 50mm long'
    },
    'C\'': {
        'Cf': 44e8,
        'FTH': 83,
        'desc': 'Base metal at toe of transverse stiffener welds'
    },
    'D': {
        'Cf': 22e8,
        'FTH': 48,
        'desc': 'Welded attachments 50-100mm long'
    },
    'E': {
        'Cf': 11e8,
        'FTH': 31,
        'desc': 'Welded attachments >100mm, cover plate ends'
    },
    'E\'': {
        'Cf': 3.9e8,
        'FTH': 18,
        'desc': 'Base metal at ends of partial-length cover plates'
    },
    'F': {
        'Cf': 150e8,
        'FTH': 55,
        'desc': 'Shear stress on weld throat'
    },
}

# ============================================================================
# IPE SECTIONS (European I-beams)
# ============================================================================
IPE = {
    'IPE 200': {'d': 200, 'bf': 100, 'tf': 8.5, 'tw': 5.6, 'r': 12, 'A': 2848, 'Ix': 19.4e6, 'Iy': 1.42e6, 'Sx': 194e3, 'Zx': 220e3, 'mass': 22.4},
    'IPE 220': {'d': 220, 'bf': 110, 'tf': 9.2, 'tw': 5.9, 'r': 12, 'A': 3337, 'Ix': 27.7e6, 'Iy': 2.05e6, 'Sx': 252e3, 'Zx': 285e3, 'mass': 26.2},
    'IPE 240': {'d': 240, 'bf': 120, 'tf': 9.8, 'tw': 6.2, 'r': 15, 'A': 3912, 'Ix': 38.9e6, 'Iy': 2.84e6, 'Sx': 324e3, 'Zx': 366e3, 'mass': 30.7},
    'IPE 270': {'d': 270, 'bf': 135, 'tf': 10.2, 'tw': 6.6, 'r': 15, 'A': 4594, 'Ix': 57.9e6, 'Iy': 4.20e6, 'Sx': 429e3, 'Zx': 484e3, 'mass': 36.1},
    'IPE 300': {'d': 300, 'bf': 150, 'tf': 10.7, 'tw': 7.1, 'r': 15, 'A': 5381, 'Ix': 83.6e6, 'Iy': 6.04e6, 'Sx': 557e3, 'Zx': 628e3, 'mass': 42.2},
    'IPE 330': {'d': 330, 'bf': 160, 'tf': 11.5, 'tw': 7.5, 'r': 18, 'A': 6261, 'Ix': 118e6, 'Iy': 7.88e6, 'Sx': 713e3, 'Zx': 804e3, 'mass': 49.1},
    'IPE 360': {'d': 360, 'bf': 170, 'tf': 12.7, 'tw': 8.0, 'r': 18, 'A': 7273, 'Ix': 163e6, 'Iy': 10.4e6, 'Sx': 904e3, 'Zx': 1019e3, 'mass': 57.1},
    'IPE 400': {'d': 400, 'bf': 180, 'tf': 13.5, 'tw': 8.6, 'r': 21, 'A': 8446, 'Ix': 231e6, 'Iy': 13.2e6, 'Sx': 1160e3, 'Zx': 1307e3, 'mass': 66.3},
    'IPE 450': {'d': 450, 'bf': 190, 'tf': 14.6, 'tw': 9.4, 'r': 21, 'A': 9882, 'Ix': 337e6, 'Iy': 16.8e6, 'Sx': 1500e3, 'Zx': 1702e3, 'mass': 77.6},
    'IPE 500': {'d': 500, 'bf': 200, 'tf': 16.0, 'tw': 10.2, 'r': 21, 'A': 11550, 'Ix': 482e6, 'Iy': 21.4e6, 'Sx': 1930e3, 'Zx': 2194e3, 'mass': 90.7},
    'IPE 550': {'d': 550, 'bf': 210, 'tf': 17.2, 'tw': 11.1, 'r': 24, 'A': 13440, 'Ix': 671e6, 'Iy': 26.7e6, 'Sx': 2440e3, 'Zx': 2787e3, 'mass': 106},
    'IPE 600': {'d': 600, 'bf': 220, 'tf': 19.0, 'tw': 12.0, 'r': 24, 'A': 15600, 'Ix': 921e6, 'Iy': 33.9e6, 'Sx': 3070e3, 'Zx': 3512e3, 'mass': 122},
}

# ============================================================================
# HEA SECTIONS (European Wide Flange - Light)
# ============================================================================
HEA = {
    'HEA 200': {'d': 190, 'bf': 200, 'tf': 10.0, 'tw': 6.5, 'r': 18, 'A': 5383, 'Ix': 36.9e6, 'Iy': 13.4e6, 'Sx': 389e3, 'Zx': 429e3, 'mass': 42.3},
    'HEA 220': {'d': 210, 'bf': 220, 'tf': 11.0, 'tw': 7.0, 'r': 18, 'A': 6434, 'Ix': 54.1e6, 'Iy': 19.5e6, 'Sx': 515e3, 'Zx': 568e3, 'mass': 50.5},
    'HEA 240': {'d': 230, 'bf': 240, 'tf': 12.0, 'tw': 7.5, 'r': 21, 'A': 7684, 'Ix': 77.6e6, 'Iy': 27.7e6, 'Sx': 675e3, 'Zx': 744e3, 'mass': 60.3},
    'HEA 260': {'d': 250, 'bf': 260, 'tf': 12.5, 'tw': 7.5, 'r': 24, 'A': 8682, 'Ix': 104e6, 'Iy': 36.7e6, 'Sx': 836e3, 'Zx': 919e3, 'mass': 68.2},
    'HEA 280': {'d': 270, 'bf': 280, 'tf': 13.0, 'tw': 8.0, 'r': 24, 'A': 9726, 'Ix': 137e6, 'Iy': 47.5e6, 'Sx': 1010e3, 'Zx': 1112e3, 'mass': 76.4},
    'HEA 300': {'d': 290, 'bf': 300, 'tf': 14.0, 'tw': 8.5, 'r': 27, 'A': 11250, 'Ix': 183e6, 'Iy': 63.1e6, 'Sx': 1260e3, 'Zx': 1383e3, 'mass': 88.3},
    'HEA 320': {'d': 310, 'bf': 300, 'tf': 15.5, 'tw': 9.0, 'r': 27, 'A': 12440, 'Ix': 229e6, 'Iy': 69.8e6, 'Sx': 1480e3, 'Zx': 1628e3, 'mass': 97.6},
    'HEA 340': {'d': 330, 'bf': 300, 'tf': 16.5, 'tw': 9.5, 'r': 27, 'A': 13330, 'Ix': 276e6, 'Iy': 74.1e6, 'Sx': 1680e3, 'Zx': 1850e3, 'mass': 105},
    'HEA 360': {'d': 350, 'bf': 300, 'tf': 17.5, 'tw': 10.0, 'r': 27, 'A': 14280, 'Ix': 331e6, 'Iy': 78.5e6, 'Sx': 1890e3, 'Zx': 2088e3, 'mass': 112},
    'HEA 400': {'d': 390, 'bf': 300, 'tf': 19.0, 'tw': 11.0, 'r': 27, 'A': 15900, 'Ix': 451e6, 'Iy': 85.6e6, 'Sx': 2310e3, 'Zx': 2562e3, 'mass': 125},
    'HEA 450': {'d': 440, 'bf': 300, 'tf': 21.0, 'tw': 11.5, 'r': 27, 'A': 17800, 'Ix': 637e6, 'Iy': 94.6e6, 'Sx': 2900e3, 'Zx': 3216e3, 'mass': 140},
    'HEA 500': {'d': 490, 'bf': 300, 'tf': 23.0, 'tw': 12.0, 'r': 27, 'A': 19800, 'Ix': 869e6, 'Iy': 104e6, 'Sx': 3550e3, 'Zx': 3949e3, 'mass': 155},
    'HEA 550': {'d': 540, 'bf': 300, 'tf': 24.0, 'tw': 12.5, 'r': 27, 'A': 21180, 'Ix': 1120e6, 'Iy': 111e6, 'Sx': 4150e3, 'Zx': 4622e3, 'mass': 166},
    'HEA 600': {'d': 590, 'bf': 300, 'tf': 25.0, 'tw': 13.0, 'r': 27, 'A': 22640, 'Ix': 1410e6, 'Iy': 117e6, 'Sx': 4790e3, 'Zx': 5350e3, 'mass': 178},
    'HEA 700': {'d': 690, 'bf': 300, 'tf': 27.0, 'tw': 14.5, 'r': 27, 'A': 26050, 'Ix': 2150e6, 'Iy': 131e6, 'Sx': 6240e3, 'Zx': 6985e3, 'mass': 204},
    'HEA 800': {'d': 790, 'bf': 300, 'tf': 28.0, 'tw': 15.0, 'r': 30, 'A': 28570, 'Ix': 3030e6, 'Iy': 137e6, 'Sx': 7680e3, 'Zx': 8699e3, 'mass': 224},
    'HEA 900': {'d': 890, 'bf': 300, 'tf': 30.0, 'tw': 16.0, 'r': 30, 'A': 32120, 'Ix': 4220e6, 'Iy': 146e6, 'Sx': 9480e3, 'Zx': 10810e3, 'mass': 252},
    'HEA 1000': {'d': 990, 'bf': 300, 'tf': 31.0, 'tw': 16.5, 'r': 30, 'A': 34680, 'Ix': 5530e6, 'Iy': 152e6, 'Sx': 11180e3, 'Zx': 12820e3, 'mass': 272},
}

# ============================================================================
# HEB SECTIONS (European Wide Flange - Medium)
# ============================================================================
HEB = {
    'HEB 200': {'d': 200, 'bf': 200, 'tf': 15.0, 'tw': 9.0, 'r': 18, 'A': 7808, 'Ix': 57.0e6, 'Iy': 20.0e6, 'Sx': 570e3, 'Zx': 642e3, 'mass': 61.3},
    'HEB 220': {'d': 220, 'bf': 220, 'tf': 16.0, 'tw': 9.5, 'r': 18, 'A': 9104, 'Ix': 80.9e6, 'Iy': 28.4e6, 'Sx': 736e3, 'Zx': 827e3, 'mass': 71.5},
    'HEB 240': {'d': 240, 'bf': 240, 'tf': 17.0, 'tw': 10.0, 'r': 21, 'A': 10600, 'Ix': 112e6, 'Iy': 39.2e6, 'Sx': 938e3, 'Zx': 1053e3, 'mass': 83.2},
    'HEB 260': {'d': 260, 'bf': 260, 'tf': 17.5, 'tw': 10.0, 'r': 24, 'A': 11840, 'Ix': 149e6, 'Iy': 51.3e6, 'Sx': 1150e3, 'Zx': 1283e3, 'mass': 93.0},
    'HEB 280': {'d': 280, 'bf': 280, 'tf': 18.0, 'tw': 10.5, 'r': 24, 'A': 13140, 'Ix': 193e6, 'Iy': 65.9e6, 'Sx': 1380e3, 'Zx': 1534e3, 'mass': 103},
    'HEB 300': {'d': 300, 'bf': 300, 'tf': 19.0, 'tw': 11.0, 'r': 27, 'A': 14910, 'Ix': 252e6, 'Iy': 85.6e6, 'Sx': 1680e3, 'Zx': 1869e3, 'mass': 117},
    'HEB 320': {'d': 320, 'bf': 300, 'tf': 20.5, 'tw': 11.5, 'r': 27, 'A': 16130, 'Ix': 308e6, 'Iy': 93.9e6, 'Sx': 1930e3, 'Zx': 2149e3, 'mass': 127},
    'HEB 340': {'d': 340, 'bf': 300, 'tf': 21.5, 'tw': 12.0, 'r': 27, 'A': 17090, 'Ix': 366e6, 'Iy': 96.9e6, 'Sx': 2160e3, 'Zx': 2408e3, 'mass': 134},
    'HEB 360': {'d': 360, 'bf': 300, 'tf': 22.5, 'tw': 12.5, 'r': 27, 'A': 18060, 'Ix': 432e6, 'Iy': 101e6, 'Sx': 2400e3, 'Zx': 2683e3, 'mass': 142},
    'HEB 400': {'d': 400, 'bf': 300, 'tf': 24.0, 'tw': 13.5, 'r': 27, 'A': 19780, 'Ix': 577e6, 'Iy': 108e6, 'Sx': 2880e3, 'Zx': 3232e3, 'mass': 155},
    'HEB 450': {'d': 450, 'bf': 300, 'tf': 26.0, 'tw': 14.0, 'r': 27, 'A': 21830, 'Ix': 799e6, 'Iy': 117e6, 'Sx': 3550e3, 'Zx': 3982e3, 'mass': 171},
    'HEB 500': {'d': 500, 'bf': 300, 'tf': 28.0, 'tw': 14.5, 'r': 27, 'A': 23860, 'Ix': 1072e6, 'Iy': 126e6, 'Sx': 4290e3, 'Zx': 4815e3, 'mass': 187},
    'HEB 550': {'d': 550, 'bf': 300, 'tf': 29.0, 'tw': 15.0, 'r': 27, 'A': 25440, 'Ix': 1367e6, 'Iy': 131e6, 'Sx': 4970e3, 'Zx': 5591e3, 'mass': 199},
    'HEB 600': {'d': 600, 'bf': 300, 'tf': 30.0, 'tw': 15.5, 'r': 27, 'A': 27000, 'Ix': 1710e6, 'Iy': 135e6, 'Sx': 5700e3, 'Zx': 6425e3, 'mass': 212},
    'HEB 700': {'d': 700, 'bf': 300, 'tf': 32.0, 'tw': 17.0, 'r': 27, 'A': 30640, 'Ix': 2569e6, 'Iy': 144e6, 'Sx': 7340e3, 'Zx': 8327e3, 'mass': 241},
    'HEB 800': {'d': 800, 'bf': 300, 'tf': 33.0, 'tw': 17.5, 'r': 30, 'A': 33430, 'Ix': 3591e6, 'Iy': 149e6, 'Sx': 8980e3, 'Zx': 10230e3, 'mass': 262},
    'HEB 900': {'d': 900, 'bf': 300, 'tf': 35.0, 'tw': 18.5, 'r': 30, 'A': 37110, 'Ix': 4941e6, 'Iy': 158e6, 'Sx': 10980e3, 'Zx': 12580e3, 'mass': 291},
    'HEB 1000': {'d': 1000, 'bf': 300, 'tf': 36.0, 'tw': 19.0, 'r': 30, 'A': 40040, 'Ix': 6444e6, 'Iy': 163e6, 'Sx': 12890e3, 'Zx': 14860e3, 'mass': 314},
}

# ============================================================================
# UB SECTIONS (British Universal Beams)
# ============================================================================
UB = {
    'UB 305x165x40': {'d': 303.4, 'bf': 165.0, 'tf': 10.2, 'tw': 6.0, 'r': 8.9, 'A': 5125, 'Ix': 85.5e6, 'Iy': 7.64e6, 'Sx': 564e3, 'Zx': 623e3, 'mass': 40.3},
    'UB 356x171x51': {'d': 355.0, 'bf': 171.5, 'tf': 11.5, 'tw': 7.4, 'r': 10.2, 'A': 6490, 'Ix': 142e6, 'Iy': 9.68e6, 'Sx': 800e3, 'Zx': 896e3, 'mass': 51.0},
    'UB 406x178x60': {'d': 406.4, 'bf': 177.9, 'tf': 12.8, 'tw': 7.9, 'r': 10.2, 'A': 7640, 'Ix': 215e6, 'Iy': 12.0e6, 'Sx': 1060e3, 'Zx': 1194e3, 'mass': 60.1},
    'UB 457x191x67': {'d': 453.4, 'bf': 189.9, 'tf': 12.7, 'tw': 8.5, 'r': 10.2, 'A': 8550, 'Ix': 294e6, 'Iy': 14.5e6, 'Sx': 1300e3, 'Zx': 1471e3, 'mass': 67.1},
    'UB 457x191x82': {'d': 460.0, 'bf': 191.3, 'tf': 16.0, 'tw': 9.9, 'r': 10.2, 'A': 10400, 'Ix': 370e6, 'Iy': 18.5e6, 'Sx': 1610e3, 'Zx': 1831e3, 'mass': 82.0},
    'UB 533x210x92': {'d': 533.1, 'bf': 209.3, 'tf': 15.6, 'tw': 10.1, 'r': 12.7, 'A': 11700, 'Ix': 554e6, 'Iy': 23.9e6, 'Sx': 2080e3, 'Zx': 2360e3, 'mass': 92.1},
    'UB 533x210x109': {'d': 539.5, 'bf': 210.8, 'tf': 18.8, 'tw': 11.6, 'r': 12.7, 'A': 13900, 'Ix': 666e6, 'Iy': 29.2e6, 'Sx': 2470e3, 'Zx': 2828e3, 'mass': 109},
    'UB 610x229x101': {'d': 602.6, 'bf': 227.6, 'tf': 14.8, 'tw': 10.5, 'r': 12.7, 'A': 12900, 'Ix': 756e6, 'Iy': 29.4e6, 'Sx': 2510e3, 'Zx': 2881e3, 'mass': 101},
    'UB 610x229x125': {'d': 612.2, 'bf': 229.0, 'tf': 19.6, 'tw': 11.9, 'r': 12.7, 'A': 15900, 'Ix': 986e6, 'Iy': 39.3e6, 'Sx': 3220e3, 'Zx': 3676e3, 'mass': 125},
    'UB 686x254x125': {'d': 677.9, 'bf': 253.0, 'tf': 16.2, 'tw': 11.7, 'r': 15.2, 'A': 15900, 'Ix': 1180e6, 'Iy': 43.9e6, 'Sx': 3480e3, 'Zx': 3994e3, 'mass': 125},
    'UB 762x267x147': {'d': 754.0, 'bf': 265.2, 'tf': 17.5, 'tw': 12.8, 'r': 16.5, 'A': 18700, 'Ix': 1690e6, 'Iy': 55.8e6, 'Sx': 4480e3, 'Zx': 5156e3, 'mass': 147},
    'UB 838x292x176': {'d': 834.9, 'bf': 291.7, 'tf': 18.8, 'tw': 14.0, 'r': 17.8, 'A': 22400, 'Ix': 2460e6, 'Iy': 78.0e6, 'Sx': 5890e3, 'Zx': 6808e3, 'mass': 176},
    'UB 914x305x201': {'d': 903.0, 'bf': 303.3, 'tf': 20.2, 'tw': 15.1, 'r': 19.1, 'A': 25600, 'Ix': 3250e6, 'Iy': 93.0e6, 'Sx': 7200e3, 'Zx': 8351e3, 'mass': 201},
    'UB 914x305x224': {'d': 910.4, 'bf': 304.1, 'tf': 23.9, 'tw': 15.9, 'r': 19.1, 'A': 28560, 'Ix': 3760e6, 'Iy': 112e6, 'Sx': 8260e3, 'Zx': 9535e3, 'mass': 224},
    'UB 914x305x253': {'d': 918.4, 'bf': 305.5, 'tf': 27.9, 'tw': 17.3, 'r': 19.1, 'A': 32250, 'Ix': 4360e6, 'Iy': 133e6, 'Sx': 9500e3, 'Zx': 10940e3, 'mass': 253},
}

# ============================================================================
# UC SECTIONS (British Universal Columns)
# ============================================================================
UC = {
    'UC 152x152x23': {'d': 152.4, 'bf': 152.2, 'tf': 6.8, 'tw': 5.8, 'r': 7.6, 'A': 2950, 'Ix': 12.5e6, 'Iy': 4.00e6, 'Sx': 164e3, 'Zx': 182e3, 'mass': 23.0},
    'UC 152x152x30': {'d': 157.6, 'bf': 152.9, 'tf': 9.4, 'tw': 6.5, 'r': 7.6, 'A': 3830, 'Ix': 17.5e6, 'Iy': 5.60e6, 'Sx': 222e3, 'Zx': 248e3, 'mass': 30.0},
    'UC 152x152x37': {'d': 161.8, 'bf': 154.4, 'tf': 11.5, 'tw': 8.0, 'r': 7.6, 'A': 4720, 'Ix': 22.2e6, 'Iy': 7.07e6, 'Sx': 274e3, 'Zx': 309e3, 'mass': 37.0},
    'UC 203x203x46': {'d': 203.2, 'bf': 203.6, 'tf': 11.0, 'tw': 7.2, 'r': 10.2, 'A': 5870, 'Ix': 45.7e6, 'Iy': 15.4e6, 'Sx': 450e3, 'Zx': 497e3, 'mass': 46.1},
    'UC 203x203x60': {'d': 209.6, 'bf': 205.8, 'tf': 14.2, 'tw': 9.4, 'r': 10.2, 'A': 7640, 'Ix': 61.2e6, 'Iy': 20.5e6, 'Sx': 584e3, 'Zx': 656e3, 'mass': 60.0},
    'UC 203x203x71': {'d': 215.8, 'bf': 206.4, 'tf': 17.3, 'tw': 10.0, 'r': 10.2, 'A': 9050, 'Ix': 76.4e6, 'Iy': 25.4e6, 'Sx': 708e3, 'Zx': 799e3, 'mass': 71.0},
    'UC 254x254x73': {'d': 254.1, 'bf': 254.6, 'tf': 14.2, 'tw': 8.6, 'r': 12.7, 'A': 9320, 'Ix': 114e6, 'Iy': 39.4e6, 'Sx': 898e3, 'Zx': 992e3, 'mass': 73.1},
    'UC 254x254x89': {'d': 260.3, 'bf': 256.3, 'tf': 17.3, 'tw': 10.3, 'r': 12.7, 'A': 11400, 'Ix': 143e6, 'Iy': 48.5e6, 'Sx': 1100e3, 'Zx': 1224e3, 'mass': 89.5},
    'UC 254x254x107': {'d': 266.7, 'bf': 258.8, 'tf': 20.5, 'tw': 12.8, 'r': 12.7, 'A': 13600, 'Ix': 175e6, 'Iy': 59.0e6, 'Sx': 1310e3, 'Zx': 1484e3, 'mass': 107},
    'UC 305x305x97': {'d': 307.9, 'bf': 305.3, 'tf': 15.4, 'tw': 9.9, 'r': 15.2, 'A': 12300, 'Ix': 222e6, 'Iy': 72.9e6, 'Sx': 1440e3, 'Zx': 1592e3, 'mass': 96.9},
    'UC 305x305x118': {'d': 314.5, 'bf': 307.4, 'tf': 18.7, 'tw': 12.0, 'r': 15.2, 'A': 15000, 'Ix': 276e6, 'Iy': 90.7e6, 'Sx': 1760e3, 'Zx': 1958e3, 'mass': 118},
    'UC 305x305x137': {'d': 320.5, 'bf': 309.2, 'tf': 21.7, 'tw': 13.8, 'r': 15.2, 'A': 17400, 'Ix': 327e6, 'Iy': 107e6, 'Sx': 2040e3, 'Zx': 2297e3, 'mass': 137},
    'UC 305x305x158': {'d': 327.1, 'bf': 311.2, 'tf': 25.0, 'tw': 15.8, 'r': 15.2, 'A': 20100, 'Ix': 387e6, 'Iy': 126e6, 'Sx': 2370e3, 'Zx': 2680e3, 'mass': 158},
}

# ============================================================================
# CHANNEL SECTIONS - UPN (European Standard Channels)
# ============================================================================
UPN = {
    'UPN 80': {'d': 80, 'bf': 45, 'tf': 8.0, 'tw': 6.0, 'A': 1100, 'Ix': 1.06e6, 'Iy': 0.194e6, 'Sx': 26.5e3, 'mass': 8.64, 'cy': 14.5},
    'UPN 100': {'d': 100, 'bf': 50, 'tf': 8.5, 'tw': 6.0, 'A': 1350, 'Ix': 2.06e6, 'Iy': 0.293e6, 'Sx': 41.2e3, 'mass': 10.6, 'cy': 15.5},
    'UPN 120': {'d': 120, 'bf': 55, 'tf': 9.0, 'tw': 7.0, 'A': 1700, 'Ix': 3.64e6, 'Iy': 0.432e6, 'Sx': 60.7e3, 'mass': 13.4, 'cy': 16.0},
    'UPN 140': {'d': 140, 'bf': 60, 'tf': 10.0, 'tw': 7.0, 'A': 2040, 'Ix': 6.05e6, 'Iy': 0.627e6, 'Sx': 86.4e3, 'mass': 16.0, 'cy': 17.5},
    'UPN 160': {'d': 160, 'bf': 65, 'tf': 10.5, 'tw': 7.5, 'A': 2400, 'Ix': 9.25e6, 'Iy': 0.853e6, 'Sx': 116e3, 'mass': 18.8, 'cy': 18.4},
    'UPN 180': {'d': 180, 'bf': 70, 'tf': 11.0, 'tw': 8.0, 'A': 2800, 'Ix': 13.5e6, 'Iy': 1.14e6, 'Sx': 150e3, 'mass': 22.0, 'cy': 19.2},
    'UPN 200': {'d': 200, 'bf': 75, 'tf': 11.5, 'tw': 8.5, 'A': 3220, 'Ix': 19.1e6, 'Iy': 1.48e6, 'Sx': 191e3, 'mass': 25.3, 'cy': 20.1},
    'UPN 220': {'d': 220, 'bf': 80, 'tf': 12.5, 'tw': 9.0, 'A': 3740, 'Ix': 27.0e6, 'Iy': 1.97e6, 'Sx': 245e3, 'mass': 29.4, 'cy': 21.2},
    'UPN 240': {'d': 240, 'bf': 85, 'tf': 13.0, 'tw': 9.5, 'A': 4230, 'Ix': 36.0e6, 'Iy': 2.48e6, 'Sx': 300e3, 'mass': 33.2, 'cy': 22.0},
    'UPN 260': {'d': 260, 'bf': 90, 'tf': 14.0, 'tw': 10.0, 'A': 4830, 'Ix': 48.0e6, 'Iy': 3.17e6, 'Sx': 369e3, 'mass': 37.9, 'cy': 23.6},
    'UPN 280': {'d': 280, 'bf': 95, 'tf': 15.0, 'tw': 10.0, 'A': 5330, 'Ix': 63.2e6, 'Iy': 3.99e6, 'Sx': 451e3, 'mass': 41.8, 'cy': 25.3},
    'UPN 300': {'d': 300, 'bf': 100, 'tf': 16.0, 'tw': 10.0, 'A': 5880, 'Ix': 80.3e6, 'Iy': 4.95e6, 'Sx': 535e3, 'mass': 46.2, 'cy': 27.0},
    'UPN 320': {'d': 320, 'bf': 100, 'tf': 17.5, 'tw': 14.0, 'A': 7580, 'Ix': 108e6, 'Iy': 5.97e6, 'Sx': 679e3, 'mass': 59.5, 'cy': 26.6},
    'UPN 350': {'d': 350, 'bf': 100, 'tf': 16.0, 'tw': 14.0, 'A': 7770, 'Ix': 128e6, 'Iy': 5.70e6, 'Sx': 734e3, 'mass': 60.6, 'cy': 25.0},
    'UPN 380': {'d': 380, 'bf': 102, 'tf': 16.5, 'tw': 13.5, 'A': 8040, 'Ix': 157e6, 'Iy': 6.15e6, 'Sx': 828e3, 'mass': 63.1, 'cy': 25.6},
    'UPN 400': {'d': 400, 'bf': 110, 'tf': 18.0, 'tw': 14.0, 'A': 9170, 'Ix': 203e6, 'Iy': 8.46e6, 'Sx': 1020e3, 'mass': 71.8, 'cy': 28.0},
}

# ============================================================================
# CHANNEL SECTIONS - PFC (Parallel Flange Channels)
# ============================================================================
PFC = {
    'PFC 100': {'d': 100, 'bf': 50, 'tf': 8.5, 'tw': 5.0, 'A': 1320, 'Ix': 2.07e6, 'Iy': 0.262e6, 'Sx': 41.4e3, 'mass': 10.4, 'cy': 14.7},
    'PFC 125': {'d': 125, 'bf': 65, 'tf': 9.5, 'tw': 5.5, 'A': 1680, 'Ix': 4.83e6, 'Iy': 0.631e6, 'Sx': 77.3e3, 'mass': 13.2, 'cy': 18.6},
    'PFC 150': {'d': 150, 'bf': 75, 'tf': 10.0, 'tw': 5.5, 'A': 2080, 'Ix': 8.19e6, 'Iy': 1.06e6, 'Sx': 109e3, 'mass': 16.3, 'cy': 21.3},
    'PFC 180': {'d': 180, 'bf': 90, 'tf': 12.5, 'tw': 6.5, 'A': 3220, 'Ix': 17.0e6, 'Iy': 2.38e6, 'Sx': 189e3, 'mass': 25.3, 'cy': 26.5},
    'PFC 200': {'d': 200, 'bf': 90, 'tf': 14.0, 'tw': 6.5, 'A': 3610, 'Ix': 23.5e6, 'Iy': 2.69e6, 'Sx': 235e3, 'mass': 28.4, 'cy': 26.1},
    'PFC 230': {'d': 230, 'bf': 90, 'tf': 14.0, 'tw': 7.5, 'A': 4210, 'Ix': 34.2e6, 'Iy': 2.87e6, 'Sx': 297e3, 'mass': 33.0, 'cy': 25.3},
    'PFC 260': {'d': 260, 'bf': 90, 'tf': 14.0, 'tw': 8.0, 'A': 4670, 'Ix': 47.4e6, 'Iy': 2.97e6, 'Sx': 365e3, 'mass': 36.6, 'cy': 24.6},
    'PFC 300': {'d': 300, 'bf': 100, 'tf': 16.5, 'tw': 9.0, 'A': 6100, 'Ix': 78.8e6, 'Iy': 4.68e6, 'Sx': 525e3, 'mass': 47.9, 'cy': 27.8},
    'PFC 380': {'d': 380, 'bf': 100, 'tf': 17.5, 'tw': 9.5, 'A': 7100, 'Ix': 143e6, 'Iy': 5.17e6, 'Sx': 752e3, 'mass': 55.7, 'cy': 26.4},
    'PFC 430': {'d': 430, 'bf': 100, 'tf': 19.0, 'tw': 11.0, 'A': 8400, 'Ix': 210e6, 'Iy': 5.68e6, 'Sx': 977e3, 'mass': 65.9, 'cy': 26.0},
}

# ============================================================================
# AGGREGATED DATABASES
# ============================================================================
SECTION_DB = {
    'IPE': IPE,
    'HEA': HEA,
    'HEB': HEB,
    'UB': UB,
    'UC': UC
}

CHANNEL_DB = {
    'UPN': UPN,
    'PFC': PFC
}



from dataclasses import dataclass, field



@dataclass
class CraneData:
    crane_id: int = 1
    capacity_tonnes: float = 10.0
    bridge_weight: float = 5.0          # tonnes
    trolley_weight: float = 0.72        # tonnes
    bridge_span: float = 15.0           # m
    min_hook_approach: float = 1.0      # m
    wheel_base: float = 2.2             # m - total wheel base (first to last wheel)
    buffer_left: float = 0.29           # m
    buffer_right: float = 0.29          # m
    num_wheels: int = 2                 # wheels per rail (2 or 4)
    impact_v: float = 0.25              # vertical impact factor
    impact_h: float = 0.20              # horizontal impact factor
    impact_l: float = 0.10              # longitudinal impact factor
    
    # 4-wheel configuration - distances between consecutive wheels
    # For 4 wheels: W1 --[d12]-- W2 --[d23]-- W3 --[d34]-- W4
    wheel_spacing_12: float = 0.0       # m - distance W1 to W2
    wheel_spacing_23: float = 0.0       # m - distance W2 to W3 (center gap)
    wheel_spacing_34: float = 0.0       # m - distance W3 to W4
    
    # Direct input option
    use_direct_input: bool = False
    direct_max_wheel_load: float = 0.0  # kN
    direct_min_wheel_load: float = 0.0  # kN
    direct_lateral_load: float = 0.0    # kN per wheel
    
    def get_wheel_positions_relative(self) -> List[float]:
        """
        Get relative wheel positions from the first wheel (W1 at 0).
        Returns list of positions in meters.
        """
        if self.num_wheels == 2:
            return [0.0, self.wheel_base]
        elif self.num_wheels == 4:
            # Use individual spacings if provided, otherwise distribute evenly
            if self.wheel_spacing_12 > 0 and self.wheel_spacing_23 > 0 and self.wheel_spacing_34 > 0:
                return [
                    0.0,
                    self.wheel_spacing_12,
                    self.wheel_spacing_12 + self.wheel_spacing_23,
                    self.wheel_spacing_12 + self.wheel_spacing_23 + self.wheel_spacing_34
                ]
            else:
                # Equal spacing fallback
                spacing = self.wheel_base / 3
                return [0.0, spacing, 2*spacing, self.wheel_base]
        else:
            # Generic equal spacing for other configurations
            if self.num_wheels <= 1:
                return [0.0]
            spacing = self.wheel_base / (self.num_wheels - 1)
            return [i * spacing for i in range(self.num_wheels)]
    
    def get_total_wheel_base(self) -> float:
        """Get total wheel base (first to last wheel)"""
        positions = self.get_wheel_positions_relative()
        return positions[-1] - positions[0] if len(positions) > 1 else 0.0
    
    def calc_wheel_loads(self) -> Tuple[float, float]:
        if self.use_direct_input and self.direct_max_wheel_load > 0:
            return self.direct_max_wheel_load, self.direct_min_wheel_load or self.direct_max_wheel_load * 0.2
        
        Lb = self.bridge_span
        e_min = self.min_hook_approach
        
        P_lift = self.capacity_tonnes * GRAVITY       # kN
        P_trolley = self.trolley_weight * GRAVITY     # kN
        P_bridge = self.bridge_weight * GRAVITY       # kN
        
        R_bridge_each = P_bridge / 2.0
        P_moving = P_lift + P_trolley
        
        # Maximum: trolley at minimum approach
        R_max = R_bridge_each + P_moving * (Lb - e_min) / Lb
        # Minimum: trolley at far end
        R_min = R_bridge_each + P_moving * e_min / Lb
        
        return R_max / self.num_wheels, R_min / self.num_wheels
    
    def get_wheel_load_with_impact(self) -> float:
        max_wl, _ = self.calc_wheel_loads()
        return max_wl * (1 + self.impact_v)
    
    def get_min_wheel_load_with_impact(self) -> float:
        _, min_wl = self.calc_wheel_loads()
        return min_wl * (1 + self.impact_v)
    
    def get_lateral_load_per_wheel(self) -> float:
        if self.use_direct_input and self.direct_lateral_load > 0:
            return self.direct_lateral_load
        
        P_lift = self.capacity_tonnes * GRAVITY
        P_trolley = self.trolley_weight * GRAVITY
        H_total = self.impact_h * (P_lift + P_trolley)
        return H_total / (2 * self.num_wheels)
    
    def get_longitudinal_force(self) -> float:
        max_wl, _ = self.calc_wheel_loads()
        return self.impact_l * max_wl * self.num_wheels


@dataclass
class Section:
    name: str = "Custom"
    sec_type: str = "hot_rolled"  # hot_rolled, built_up
    
    # Basic dimensions (mm)
    d: float = 0          # Total depth
    bf: float = 0         # Flange width (symmetric) or top flange
    tf: float = 0         # Flange thickness (symmetric) or top flange
    tw: float = 0         # Web thickness
    r: float = 0          # Fillet radius
    
    # Built-up section specific (mm)
    bf_top: float = 0
    tf_top: float = 0
    bf_bot: float = 0
    tf_bot: float = 0
    hw: float = 0         # Clear web height
    
    # Cap channel
    has_cap: bool = False
    cap_name: str = ""
    cap_A: float = 0      # mm²
    cap_Ix: float = 0     # mm⁴
    cap_Iy: float = 0     # mm⁴
    cap_d: float = 0      # mm - channel depth
    cap_cy: float = 0     # mm - centroid from back of channel
    
    # Section properties
    A: float = 0          # Area (mm²)
    Ix: float = 0         # Moment of inertia x (mm⁴)
    Iy: float = 0         # Moment of inertia y (mm⁴)
    Sx: float = 0         # Section modulus x (mm³)
    Sy: float = 0         # Section modulus y (mm³)
    Zx: float = 0         # Plastic section modulus (mm³)
    rx: float = 0         # Radius of gyration x (mm)
    ry: float = 0         # Radius of gyration y (mm)
    rts: float = 0        # Effective radius for LTB (mm)
    J: float = 0          # Torsional constant (mm⁴)
    Cw: float = 0         # Warping constant (mm⁶)
    ho: float = 0         # Flange centroid distance (mm)
    y_bar: float = 0      # Centroid from bottom (mm)
    mass: float = 0       # Mass per meter (kg/m)
    
    def calc_props(self) -> 'Section':
        
        if self.sec_type == 'built_up':
            return self._calc_built_up_props()
        else:
            return self._calc_hot_rolled_props()
    
    def _calc_built_up_props(self) -> 'Section':
        # Set defaults
        if self.bf_top == 0:
            self.bf_top = self.bf
            self.tf_top = self.tf
            self.bf_bot = self.bf
            self.tf_bot = self.tf
        
        if self.hw == 0:
            self.hw = self.d - self.tf_top - self.tf_bot
        
        # Areas
        A_tf = self.bf_top * self.tf_top
        A_bf = self.bf_bot * self.tf_bot
        A_w = self.hw * self.tw
        A_I = A_tf + A_bf + A_w
        
        # Centroids from bottom
        y_tf = self.tf_bot + self.hw + self.tf_top / 2
        y_bf = self.tf_bot / 2
        y_w = self.tf_bot + self.hw / 2
        
        # With or without cap channel
        if self.has_cap and self.cap_A > 0:
            y_cap = self.d + self.cap_cy
            self.A = A_I + self.cap_A
            self.y_bar = (A_tf*y_tf + A_bf*y_bf + A_w*y_w + self.cap_A*y_cap) / self.A
        else:
            self.A = A_I
            self.y_bar = (A_tf*y_tf + A_bf*y_bf + A_w*y_w) / max(A_I, 1)
        
        # Moment of inertia
        self.ho = self.d - (self.tf_top + self.tf_bot) / 2
        
        I_tf = self.bf_top * self.tf_top**3 / 12 + A_tf * (y_tf - self.y_bar)**2
        I_bf = self.bf_bot * self.tf_bot**3 / 12 + A_bf * (y_bf - self.y_bar)**2
        I_w = self.tw * self.hw**3 / 12 + A_w * (y_w - self.y_bar)**2
        
        if self.has_cap and self.cap_A > 0:
            y_cap = self.d + self.cap_cy
            I_cap = self.cap_Ix + self.cap_A * (y_cap - self.y_bar)**2
            self.Ix = I_tf + I_bf + I_w + I_cap
            self.Iy = (self.tf_top * self.bf_top**3 + self.tf_bot * self.bf_bot**3 + 
                      self.hw * self.tw**3) / 12 + self.cap_Iy
        else:
            self.Ix = I_tf + I_bf + I_w
            self.Iy = (self.tf_top * self.bf_top**3 + self.tf_bot * self.bf_bot**3 + 
                      self.hw * self.tw**3) / 12
        
        # Section moduli
        c_top = (self.d + self.cap_d if self.has_cap else self.d) - self.y_bar
        c_bot = self.y_bar
        self.Sx = self.Ix / max(c_top, c_bot)
        self.Sy = self.Iy / max(self.bf_top / 2, self.bf_bot / 2, 1)
        
        # Torsional properties
        self.J = (self.bf_top * self.tf_top**3 + self.bf_bot * self.tf_bot**3 + 
                 self.hw * self.tw**3) / 3
        self.Cw = self.Iy * self.ho**2 / 4 if self.ho > 0 else 1
        
        self._calc_common_props()
        return self
    
    def _calc_hot_rolled_props(self) -> 'Section':
        if self.hw == 0:
            self.hw = self.d - 2 * self.tf
        if self.ho == 0:
            self.ho = self.d - self.tf
        
        # Set flange values for consistency
        if self.bf_top == 0:
            self.bf_top = self.bf
            self.tf_top = self.tf
            self.bf_bot = self.bf
            self.tf_bot = self.tf
        
        if self.has_cap and self.cap_A > 0:
            # Add cap channel to hot-rolled section
            A_I = self.A if self.A > 0 else (2 * self.bf * self.tf + self.hw * self.tw)
            y_I = self.d / 2
            y_cap = self.d + self.cap_cy
            
            total_A = A_I + self.cap_A
            self.y_bar = (A_I * y_I + self.cap_A * y_cap) / total_A
            
            I_I = self.Ix if self.Ix > 0 else (self.bf * self.d**3 / 12 - 
                  (self.bf - self.tw) * self.hw**3 / 12)
            I_cap = self.cap_Ix + self.cap_A * (y_cap - self.y_bar)**2
            I_I_shifted = I_I + A_I * (y_I - self.y_bar)**2
            
            self.Ix = I_I_shifted + I_cap
            self.A = total_A
            
            if self.Iy > 0:
                self.Iy = self.Iy + self.cap_Iy
            
            c_top = (self.d + self.cap_d) - self.y_bar
            c_bot = self.y_bar
            self.Sx = self.Ix / max(c_top, c_bot)
        
        self._calc_common_props()
        return self
    
    def _calc_common_props(self):
        if self.A > 0:
            self.rx = math.sqrt(self.Ix / self.A) if self.Ix > 0 else 0
            self.ry = math.sqrt(self.Iy / self.A) if self.Iy > 0 else 0
        
        if self.Zx == 0 and self.Sx > 0:
            self.Zx = self.Sx * 1.12  # Approximate
        
        if self.J == 0:
            self.J = (2 * self.bf * self.tf**3 + self.hw * self.tw**3) / 3
        
        if self.Cw == 0 and self.ho > 0 and self.Iy > 0:
            self.Cw = self.Iy * self.ho**2 / 4
        
        if self.rts == 0 and self.Sx > 0 and self.Iy > 0 and self.Cw > 0:
            self.rts = math.sqrt(math.sqrt(self.Iy * self.Cw) / self.Sx)
        
        if self.mass == 0 and self.A > 0:
            self.mass = self.A * RHO_STEEL / 1e6
            if self.has_cap and self.cap_A > 0:
                self.mass += self.cap_A * RHO_STEEL / 1e6


@dataclass
class StiffenerData:
    # Transverse stiffeners
    has_transverse: bool = False
    trans_spacing: float = 0      # mm
    trans_t: float = 10           # mm - thickness
    trans_b: float = 80           # mm - width (single side)
    
    # Bearing stiffeners
    has_bearing: bool = False
    bearing_t: float = 12         # mm
    bearing_b: float = 100        # mm (single side)
    
    # Longitudinal stiffeners
    has_longitudinal: bool = False
    long_t: float = 10            # mm
    long_b: float = 80            # mm
    long_position: float = 0.2    # fraction of hw from compression flange
    
    # Weld
    weld_size: float = 6          # mm


@dataclass
class LoadCaseResult:
    step_position: float
    wheel_positions: List[float]
    wheel_loads: List[float]
    R_A: float
    R_B: float
    M_max: float
    M_max_location: float
    V_max: float
    moments_at_wheels: List[float]


@dataclass
class CriticalResults:
    # Maximum moment case
    M_max: float = 0.0
    M_max_position: float = 0.0
    M_max_location: float = 0.0
    M_max_R_A: float = 0.0
    M_max_R_B: float = 0.0
    M_max_wheel_positions: List[float] = field(default_factory=list)
    
    # Maximum left reaction case
    R_A_max: float = 0.0
    R_A_max_position: float = 0.0
    R_A_max_wheel_positions: List[float] = field(default_factory=list)
    R_A_max_moment: float = 0.0
    R_A_max_R_B: float = 0.0
    
    # Maximum right reaction case
    R_B_max: float = 0.0
    R_B_max_position: float = 0.0
    R_B_max_wheel_positions: List[float] = field(default_factory=list)
    R_B_max_moment: float = 0.0
    R_B_max_R_A: float = 0.0
    
    # Minimum reactions
    R_A_min: float = 0.0
    R_B_min: float = 0.0
    
    # Maximum shear
    V_max: float = 0.0
    
    # All stepping results
    all_results: List[LoadCaseResult] = field(default_factory=list)



import numpy as np



def get_crane_group_geometry(cranes: List[CraneData]) -> Tuple[List[float], List[float], float]:
    """
    Get wheel positions relative to first wheel for a crane group.
    For multiple cranes, uses MINIMUM gap (buffer to buffer only).
    This allows cranes to be as close as physically possible.
    Supports 4-wheel cranes with variable spacing between wheels.
    """
    rel_positions = []
    wheel_loads = []
    current_pos = 0.0
    
    for i, crane in enumerate(cranes):
        Pv = crane.get_wheel_load_with_impact()
        
        # Get wheel positions relative to first wheel of THIS crane
        crane_wheel_positions = crane.get_wheel_positions_relative()
        
        # Add wheels for this crane
        for w_pos in crane_wheel_positions:
            rel_positions.append(current_pos + w_pos)
            wheel_loads.append(Pv)
        
        # MINIMUM gap to next crane = only buffer clearances
        # This is the closest cranes can physically get
        if i < len(cranes) - 1:
            # Gap = current crane's buffer_right + next crane's buffer_left
            # This represents end-stop to end-stop clearance
            gap = crane.buffer_right + cranes[i + 1].buffer_left
            crane_total_length = crane.get_total_wheel_base()
            current_pos += crane_total_length + gap
    
    total_length = rel_positions[-1] - rel_positions[0] if rel_positions else 0.0
    
    return rel_positions, wheel_loads, total_length


def analyze_single_position(L: float, first_wheel_pos: float, 
                           rel_positions: List[float],
                           wheel_loads: List[float]) -> LoadCaseResult:
    # Absolute wheel positions
    abs_positions = [first_wheel_pos + rel for rel in rel_positions]
    
    # Calculate reactions using influence lines
    # R_A influence ordinate at x = (L - x) / L
    # R_B influence ordinate at x = x / L
    R_A = sum(P * (L - x) / L for P, x in zip(wheel_loads, abs_positions))
    R_B = sum(P * x / L for P, x in zip(wheel_loads, abs_positions))
    
    # Calculate moments at each wheel position
    moments_at_wheels = []
    for i, x_i in enumerate(abs_positions):
        M = R_A * x_i
        for j, (x_j, P_j) in enumerate(zip(abs_positions, wheel_loads)):
            if x_j < x_i:
                M -= P_j * (x_i - x_j)
        moments_at_wheels.append(M)
    
    # Also check midspan
    x_mid = L / 2
    M_mid = R_A * x_mid
    for x_j, P_j in zip(abs_positions, wheel_loads):
        if x_j < x_mid:
            M_mid -= P_j * (x_mid - x_j)
    
    # Find maximum moment
    all_moments = moments_at_wheels + [M_mid]
    all_locations = abs_positions + [x_mid]
    max_idx = np.argmax(all_moments)
    
    return LoadCaseResult(
        step_position=first_wheel_pos,
        wheel_positions=abs_positions,
        wheel_loads=wheel_loads.copy(),
        R_A=R_A,
        R_B=R_B,
        M_max=all_moments[max_idx],
        M_max_location=all_locations[max_idx],
        V_max=max(R_A, R_B),
        moments_at_wheels=moments_at_wheels
    )


def run_moving_load_analysis(beam_span: float, cranes: List[CraneData],
                             step_size: float = 0.5,
                             num_cranes_on_beam: int = None) -> CriticalResults:
    L = beam_span
    
    if num_cranes_on_beam is None:
        num_cranes_on_beam = len(cranes)
    
    active_cranes = cranes[:num_cranes_on_beam]
    
    # Get crane group geometry
    rel_positions, wheel_loads, group_length = get_crane_group_geometry(active_cranes)
    
    if not rel_positions:
        return CriticalResults()
    
    # Determine travel limits
    buffer_left = active_cranes[0].buffer_left
    buffer_right = active_cranes[-1].buffer_right
    
    start_pos = buffer_left
    end_pos = L - buffer_right - group_length
    
    if end_pos < start_pos:
        end_pos = start_pos
    
    # Run stepping analysis
    results = CriticalResults()
    all_results = []
    
    current_pos = start_pos
    while current_pos <= end_pos + 0.001:
        result = analyze_single_position(L, current_pos, rel_positions, wheel_loads)
        all_results.append(result)
        current_pos += step_size
    
    # Extract critical values
    if all_results:
        # Maximum moment
        idx_M = max(range(len(all_results)), key=lambda i: all_results[i].M_max)
        results.M_max = all_results[idx_M].M_max
        results.M_max_position = all_results[idx_M].step_position
        results.M_max_location = all_results[idx_M].M_max_location
        results.M_max_R_A = all_results[idx_M].R_A
        results.M_max_R_B = all_results[idx_M].R_B
        results.M_max_wheel_positions = all_results[idx_M].wheel_positions.copy()
        
        # Maximum left reaction
        idx_RA = max(range(len(all_results)), key=lambda i: all_results[i].R_A)
        results.R_A_max = all_results[idx_RA].R_A
        results.R_A_max_position = all_results[idx_RA].step_position
        results.R_A_max_wheel_positions = all_results[idx_RA].wheel_positions.copy()
        results.R_A_max_moment = all_results[idx_RA].M_max
        results.R_A_max_R_B = all_results[idx_RA].R_B
        
        # Maximum right reaction
        idx_RB = max(range(len(all_results)), key=lambda i: all_results[i].R_B)
        results.R_B_max = all_results[idx_RB].R_B
        results.R_B_max_position = all_results[idx_RB].step_position
        results.R_B_max_wheel_positions = all_results[idx_RB].wheel_positions.copy()
        results.R_B_max_moment = all_results[idx_RB].M_max
        results.R_B_max_R_A = all_results[idx_RB].R_A
        
        # Maximum shear
        results.V_max = max(results.R_A_max, results.R_B_max)
        
        # Minimum reactions
        idx_RA_min = min(range(len(all_results)), key=lambda i: all_results[i].R_A)
        idx_RB_min = min(range(len(all_results)), key=lambda i: all_results[i].R_B)
        results.R_A_min = all_results[idx_RA_min].R_A
        results.R_B_min = all_results[idx_RB_min].R_B
        
        results.all_results = all_results
    
    return results


def calc_deflection(wheel_loads: List[float], wheel_positions: List[float],
                   L: float, E: float, I: float) -> float:
    def point_load_deflection(P, a, x, L, E, I):
        b = L - a
        if x <= a:
            return P * b * x * (L**2 - b**2 - x**2) / (6 * E * I * L)
        else:
            return P * a * (L - x) * (2 * L * x - x**2 - a**2) / (6 * E * I * L)
    
    # Convert to mm units
    L_mm = L * 1000
    positions_mm = [p * 1000 for p in wheel_positions]
    loads_N = [P * 1000 for P in wheel_loads]  # kN to N
    
    # Check at midspan and wheel positions
    check_points = [L_mm / 2] + positions_mm
    max_def = 0
    
    for x in check_points:
        if 0 < x < L_mm:
            delta = sum(point_load_deflection(P, a, x, L_mm, E, I)
                       for P, a in zip(loads_N, positions_mm))
            max_def = max(max_def, abs(delta))
    
    return max_def


def check_compactness(sec: Section, Fy: float) -> Dict:
    bf = max(sec.bf_top, sec.bf_bot) if sec.bf_top > 0 else sec.bf
    tf = min(sec.tf_top, sec.tf_bot) if sec.tf_top > 0 else sec.tf
    hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
    
    lambda_f = bf / (2 * tf) if tf > 0 else 999
    lambda_pf = 0.38 * math.sqrt(E_STEEL / Fy)
    lambda_rf = 1.0 * math.sqrt(E_STEEL / Fy)
    
    if lambda_f <= lambda_pf:
        flange_class = "Compact"
    elif lambda_f <= lambda_rf:
        flange_class = "Noncompact"
    else:
        flange_class = "Slender"
    
    lambda_w = hw / sec.tw if sec.tw > 0 else 999
    lambda_pw = 3.76 * math.sqrt(E_STEEL / Fy)
    lambda_rw = 5.70 * math.sqrt(E_STEEL / Fy)
    
    if lambda_w <= lambda_pw:
        web_class = "Compact"
    elif lambda_w <= lambda_rw:
        web_class = "Noncompact"
    else:
        web_class = "Slender"
    
    return {
        'lambda_f': lambda_f, 'lambda_pf': lambda_pf, 'lambda_rf': lambda_rf,
        'flange_class': flange_class, 'lambda_w': lambda_w, 'lambda_pw': lambda_pw,
        'lambda_rw': lambda_rw, 'web_class': web_class,
        'compact': flange_class == "Compact" and web_class == "Compact"
    }


def calc_Lp_Lr(sec: Section, Fy: float) -> tuple:
    ry = sec.ry if sec.ry > 0 else math.sqrt(sec.Iy / sec.A) if sec.A > 0 else 1
    Lp = 1.76 * ry * math.sqrt(E_STEEL / Fy)
    
    ho = sec.ho if sec.ho > 0 else sec.d - sec.tf
    rts = sec.rts if sec.rts > 0 else ry
    c = 1.0
    
    if sec.Sx > 0 and ho > 0:
        term1 = sec.J * c / (sec.Sx * ho)
        term2 = math.sqrt(term1**2 + 6.76 * (0.7 * Fy / E_STEEL)**2)
        Lr = 1.95 * rts * (E_STEEL / (0.7 * Fy)) * math.sqrt(term1 + term2)
    else:
        Lr = Lp * 3
    
    return Lp, Lr


def calc_flexural_strength(sec: Section, Fy: float, Lb: float, cmp: Dict) -> Dict:
    Lp, Lr = calc_Lp_Lr(sec, Fy)
    
    Mp = Fy * sec.Zx / 1e6 if sec.Zx > 0 else Fy * sec.Sx * 1.12 / 1e6
    My = Fy * sec.Sx / 1e6
    Cb = 1.0
    
    if Lb <= Lp:
        Mn = Mp
        limit_state = "Yielding (Lb ≤ Lp)"
    elif Lb <= Lr:
        Mn = Cb * (Mp - (Mp - 0.7 * My) * (Lb - Lp) / (Lr - Lp))
        Mn = min(Mn, Mp)
        limit_state = "Inelastic LTB"
    else:
        ho = sec.ho if sec.ho > 0 else sec.d - sec.tf
        rts = sec.rts if sec.rts > 0 else sec.ry
        c = 1.0
        if sec.Sx > 0 and ho > 0:
            term = sec.J * c / (sec.Sx * ho)
            Fcr = (Cb * math.pi**2 * E_STEEL / (Lb / rts)**2) * \
                  math.sqrt(1 + 0.078 * term * (Lb / rts)**2)
        else:
            Fcr = 0.7 * Fy
        Mn = Fcr * sec.Sx / 1e6
        Mn = min(Mn, Mp)
        limit_state = "Elastic LTB"
    
    # Flange local buckling
    if cmp['flange_class'] == "Noncompact":
        Mn_flb = Mp - (Mp - 0.7 * My) * (cmp['lambda_f'] - cmp['lambda_pf']) / \
                 (cmp['lambda_rf'] - cmp['lambda_pf'])
        if Mn_flb < Mn:
            Mn = Mn_flb
            limit_state = "Flange Local Buckling"
    
    # Slender web - Rpg factor
    if cmp['web_class'] == "Slender":
        bf = max(sec.bf_top, sec.bf_bot) if sec.bf_top > 0 else sec.bf
        tf = max(sec.tf_top, sec.tf_bot) if sec.tf_top > 0 else sec.tf
        aw = sec.hw * sec.tw / (bf * tf)
        Rpg = 1 - aw / (1200 + 300 * aw) * (sec.hw / sec.tw - 5.7 * math.sqrt(E_STEEL / Fy))
        Rpg = min(1.0, max(Rpg, 0.5))
        Mn = Mn * Rpg
        limit_state += f" + Rpg={Rpg:.3f}"
    
    return {'Mp': Mp, 'My': My, 'Lp': Lp, 'Lr': Lr, 'Lb': Lb, 'Cb': Cb,
            'Mn': Mn, 'Mn_allow': Mn / OMEGA_FLEX, 'limit_state': limit_state}


def calc_shear_strength(sec: Section, Fy: float, has_stiff: bool = False,
                        stiff_spa: float = 0, use_tfa: bool = False) -> Dict:
    hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
    Aw = hw * sec.tw
    h_tw = hw / sec.tw if sec.tw > 0 else 999
    
    kv = 5.34
    if has_stiff and stiff_spa > 0 and hw > 0:
        a_h = stiff_spa / hw
        if a_h <= 3:
            kv = 5 + 5 / (a_h**2)
    
    limit1 = 1.10 * math.sqrt(kv * E_STEEL / Fy)
    limit2 = 1.37 * math.sqrt(kv * E_STEEL / Fy)
    
    if h_tw <= limit1:
        Cv1 = 1.0
        limit_state = "Shear Yielding"
    elif h_tw <= limit2:
        Cv1 = limit1 / h_tw
        limit_state = "Inelastic Buckling"
    else:
        Cv1 = 1.51 * kv * E_STEEL / (h_tw**2 * Fy)
        limit_state = "Elastic Buckling"
    
    Vn = 0.6 * Fy * Aw * Cv1 / 1000
    omega = OMEGA_SHEAR
    
    if use_tfa and has_stiff and stiff_spa > 0 and Cv1 < 1.0:
        a_h = stiff_spa / hw
        if 0.5 <= a_h <= 3:
            Cv2 = limit1 / h_tw if h_tw > limit1 else 1.0
            Vn_tfa = 0.6 * Fy * Aw * (Cv2 + (1 - Cv2) / (1.15 * math.sqrt(1 + a_h**2))) / 1000
            if Vn_tfa > Vn:
                Vn = Vn_tfa
                limit_state = "Tension Field Action"
                omega = OMEGA_SHEAR_TFA
    
    return {'Aw': Aw, 'h_tw': h_tw, 'kv': kv, 'Cv1': Cv1, 'Vn': Vn,
            'Vn_allow': Vn / omega, 'limit_state': limit_state}


def check_web_local_yielding(sec: Section, Fy: float, R: float, lb: float,
                              at_support: bool = True) -> Dict:
    k = sec.tf + sec.r if sec.r > 0 else sec.tf * 1.5
    
    if at_support:
        Rn = Fy * sec.tw * (2.5 * k + lb) / 1000
    else:
        Rn = Fy * sec.tw * (5 * k + lb) / 1000
    
    Rn_allow = Rn / OMEGA_WLY
    ratio = R / Rn_allow if Rn_allow > 0 else 999
    
    return {'k': k, 'lb': lb, 'Rn': Rn, 'Rn_allow': Rn_allow, 'R_applied': R,
            'ratio': ratio, 'status': 'OK' if ratio <= 1.0 else 'NG'}


def check_web_crippling(sec: Section, Fy: float, R: float, lb: float,
                         at_support: bool = True) -> Dict:
    tf = min(sec.tf_top, sec.tf_bot) if sec.tf_top > 0 else sec.tf
    
    if at_support:
        if lb / sec.d <= 0.2:
            Rn = 0.40 * sec.tw**2 * (1 + 3 * (lb / sec.d) * (sec.tw / tf)**1.5) * \
                 math.sqrt(E_STEEL * Fy * tf / sec.tw) / 1000
        else:
            Rn = 0.40 * sec.tw**2 * (1 + (4 * lb / sec.d - 0.2) * (sec.tw / tf)**1.5) * \
                 math.sqrt(E_STEEL * Fy * tf / sec.tw) / 1000
    else:
        Rn = 0.80 * sec.tw**2 * (1 + 3 * (lb / sec.d) * (sec.tw / tf)**1.5) * \
             math.sqrt(E_STEEL * Fy * tf / sec.tw) / 1000
    
    Rn_allow = Rn / OMEGA_WCR
    ratio = R / Rn_allow if Rn_allow > 0 else 999
    
    return {'lb': lb, 'Rn': Rn, 'Rn_allow': Rn_allow, 'R_applied': R,
            'ratio': ratio, 'status': 'OK' if ratio <= 1.0 else 'NG'}


def check_fatigue(sec: Section, M_range: float, N_cycles: int, category: str = 'E') -> Dict:
    cat = FATIGUE_CATEGORIES.get(category, FATIGUE_CATEGORIES['E'])
    Cf = cat['Cf']
    FTH = cat['FTH']
    
    f_sr = M_range * 1e6 / sec.Sx if sec.Sx > 0 else 0
    F_sr = (Cf / N_cycles) ** (1/3)
    F_sr = max(F_sr, FTH)
    
    ratio = f_sr / F_sr if F_sr > 0 else 999
    
    return {'f_sr': f_sr, 'F_sr': F_sr, 'FTH': FTH, 'Cf': Cf,
            'N_cycles': N_cycles, 'category': category,
            'ratio': ratio, 'status': 'OK' if ratio <= 1.0 else 'NG'}


# ============================================================================
# WELD DESIGN FUNCTIONS (AISC 360-16 Chapter J2 & AWS D1.1)
# ============================================================================

@dataclass
class WeldDesignData:
    """Data class for weld design parameters"""
    weld_type: str = 'fillet'           # 'fillet', 'CJP', 'PJP'
    electrode: str = 'E70'              # Electrode classification
    weld_size: float = 6.0              # mm - leg size for fillet, throat for groove
    weld_length: float = 0.0            # mm - if 0, continuous weld assumed
    num_welds: int = 2                  # Number of weld lines (typically 2 for I-beam)
    is_continuous: bool = True          # Continuous or intermittent
    intermittent_length: float = 0.0    # mm - length of each weld segment
    intermittent_spacing: float = 0.0   # mm - center-to-center spacing


def get_min_fillet_weld_size(t_thick: float) -> float:
    """Get minimum fillet weld size per AISC 360-16 Table J2.4"""
    for t_limit, w_min in sorted(MIN_FILLET_WELD_SIZE.items()):
        if t_thick <= t_limit:
            return w_min
    return 13


def get_max_fillet_weld_size(t_edge: float) -> float:
    """Get maximum fillet weld size per AISC 360-16 J2.2b"""
    if t_edge < 6.35:
        return t_edge
    else:
        return t_edge - 1.6


def calc_fillet_weld_strength(weld_size: float, FEXX: float, Fy_base: float, 
                               t_base: float, weld_angle: float = 90.0) -> Dict:
    """Calculate fillet weld strength per AISC 360-16 J2.4"""
    throat = 0.707 * weld_size
    Fnw = 0.60 * FEXX
    theta_rad = math.radians(weld_angle)
    dir_factor = 1.0 + 0.50 * (math.sin(theta_rad) ** 1.5)
    Fnw_dir = 0.60 * FEXX * dir_factor
    Rn_weld = Fnw * throat
    Rn_weld_dir = Fnw_dir * throat
    Fu_base = Fy_base * 1.3
    Rn_base = 0.60 * Fu_base * t_base
    Rn = min(Rn_weld, Rn_base)
    Rn_dir = min(Rn_weld_dir, Rn_base)
    omega = 2.00
    Rn_allow = Rn / omega
    Rn_allow_dir = Rn_dir / omega
    
    return {
        'throat': throat, 'Fnw': Fnw, 'Fnw_dir': Fnw_dir, 'dir_factor': dir_factor,
        'Rn_weld': Rn_weld, 'Rn_weld_dir': Rn_weld_dir, 'Rn_base': Rn_base,
        'Rn': Rn, 'Rn_dir': Rn_dir, 'Rn_allow': Rn_allow, 'Rn_allow_dir': Rn_allow_dir,
        'omega': omega, 'governs': 'Weld' if Rn_weld <= Rn_base else 'Base Metal'
    }


def calc_groove_weld_strength(weld_type: str, FEXX: float, Fy_base: float,
                               t_base: float, throat: float = 0) -> Dict:
    """Calculate groove weld strength per AISC 360-16 J2.3"""
    Fu_base = Fy_base * 1.3
    
    if weld_type == 'CJP':
        Rn_tension = Fy_base * t_base
        Rn_shear_weld = 0.60 * FEXX * t_base
        Rn_shear_base = 0.60 * Fy_base * t_base
        Rn_shear = min(Rn_shear_weld, Rn_shear_base)
        omega_tension = 1.67
        omega_shear = 1.50
        return {
            'throat': t_base, 'Rn_tension': Rn_tension, 'Rn_shear': Rn_shear,
            'Rn_allow_tension': Rn_tension / omega_tension,
            'Rn_allow_shear': Rn_shear / omega_shear,
            'omega_tension': omega_tension, 'omega_shear': omega_shear, 'full_strength': True
        }
    else:  # PJP
        effective_throat = throat if throat > 0 else t_base * 0.5
        Fn_weld = 0.60 * FEXX
        Rn_tension_weld = Fn_weld * effective_throat
        Rn_tension_base = Fy_base * effective_throat
        Rn_tension = min(Rn_tension_weld, Rn_tension_base)
        Rn_shear = 0.60 * FEXX * effective_throat
        omega = 2.00
        return {
            'throat': effective_throat, 'Rn_tension': Rn_tension, 'Rn_shear': Rn_shear,
            'Rn_allow_tension': Rn_tension / omega, 'Rn_allow_shear': Rn_shear / omega,
            'omega': omega, 'full_strength': False
        }


def check_weld_for_built_up_section(sec: Section, V_design: float, M_design: float,
                                     weld_data: WeldDesignData, Fy: float) -> Dict:
    """
    Check welds connecting flanges to web in built-up I-sections
    per AISC 360-16 Chapter J2 and AWS D1.1
    """
    results = {
        'weld_type': weld_data.weld_type, 'electrode': weld_data.electrode,
        'weld_size': weld_data.weld_size, 'checks': [], 'ok': True, 'ratio': 0, 'status': 'OK'
    }
    
    FEXX = WELD_ELECTRODES.get(weld_data.electrode, WELD_ELECTRODES['E70'])['FEXX']
    results['FEXX'] = FEXX
    
    # Section properties
    tf = sec.tf_top if sec.tf_top > 0 else sec.tf
    bf = sec.bf_top if sec.bf_top > 0 else sec.bf
    d = sec.d
    hw = sec.hw if sec.hw > 0 else d - 2 * tf
    tw = sec.tw
    Ix = sec.Ix
    
    # First moment of area of flange
    y_flange = (d - tf) / 2
    A_flange = bf * tf
    Q = A_flange * y_flange
    
    results['Q'] = Q
    results['y_flange'] = y_flange
    results['A_flange'] = A_flange
    
    # Shear flow
    V_N = V_design * 1000
    q = V_N * Q / Ix
    results['shear_flow'] = q
    
    n_welds = weld_data.num_welds
    q_per_weld = q / n_welds
    results['q_per_weld'] = q_per_weld
    
    # CHECK 1: Minimum Weld Size
    t_thick = max(tf, tw)
    w_min = get_min_fillet_weld_size(t_thick)
    check_min = {
        'name': 'Minimum Weld Size (Table J2.4)',
        'demand': f'{weld_data.weld_size:.1f} mm',
        'capacity': f'≥ {w_min:.1f} mm',
        'ok': weld_data.weld_size >= w_min,
        'reference': 'AISC 360-16 Table J2.4'
    }
    results['checks'].append(check_min)
    results['w_min'] = w_min
    
    # CHECK 2: Maximum Weld Size
    w_max = get_max_fillet_weld_size(min(tf, tw))
    check_max = {
        'name': 'Maximum Weld Size (J2.2b)',
        'demand': f'{weld_data.weld_size:.1f} mm',
        'capacity': f'≤ {w_max:.1f} mm',
        'ok': weld_data.weld_size <= w_max,
        'reference': 'AISC 360-16 J2.2b'
    }
    results['checks'].append(check_max)
    results['w_max'] = w_max
    
    # CHECK 3: Weld Strength
    if weld_data.weld_type == 'fillet':
        weld_str = calc_fillet_weld_strength(weld_data.weld_size, FEXX, Fy, tw, weld_angle=0)
        Rn_allow = weld_str['Rn_allow']
        ratio_strength = q_per_weld / Rn_allow if Rn_allow > 0 else 999
        check_strength = {
            'name': 'Weld Shear Strength (J2.4)',
            'demand': f'{q_per_weld:.2f} N/mm',
            'capacity': f'{Rn_allow:.2f} N/mm',
            'ratio': ratio_strength,
            'ok': ratio_strength <= 1.0,
            'reference': 'AISC 360-16 Eq. J2-4'
        }
        results['checks'].append(check_strength)
        results['throat'] = weld_str['throat']
        results['Rn_allow'] = Rn_allow
        results['weld_strength'] = weld_str
    else:
        throat = weld_data.weld_size if weld_data.weld_type == 'PJP' else tw
        weld_str = calc_groove_weld_strength(weld_data.weld_type, FEXX, Fy, tw, throat)
        Rn_allow = weld_str['Rn_allow_shear']
        ratio_strength = q_per_weld / Rn_allow if Rn_allow > 0 else 999
        check_strength = {
            'name': f'{weld_data.weld_type} Groove Weld Shear (J2.3)',
            'demand': f'{q_per_weld:.2f} N/mm',
            'capacity': f'{Rn_allow:.2f} N/mm',
            'ratio': ratio_strength,
            'ok': ratio_strength <= 1.0,
            'reference': 'AISC 360-16 Table J2.3'
        }
        results['checks'].append(check_strength)
        results['throat'] = weld_str['throat']
        results['Rn_allow'] = Rn_allow
        results['weld_strength'] = weld_str
    
    # CHECK 4: Intermittent Weld Spacing
    if not weld_data.is_continuous and weld_data.intermittent_spacing > 0:
        max_spacing = min(24 * tw, 305)
        check_spacing = {
            'name': 'Intermittent Weld Spacing',
            'demand': f'{weld_data.intermittent_spacing:.0f} mm',
            'capacity': f'≤ {max_spacing:.0f} mm',
            'ok': weld_data.intermittent_spacing <= max_spacing,
            'reference': 'AISC 360-16 J2.2b'
        }
        results['checks'].append(check_spacing)
        
        min_length = max(4 * weld_data.weld_size, 38)
        check_length = {
            'name': 'Minimum Intermittent Weld Length',
            'demand': f'{weld_data.intermittent_length:.0f} mm',
            'capacity': f'≥ {min_length:.0f} mm',
            'ok': weld_data.intermittent_length >= min_length,
            'reference': 'AISC 360-16 J2.2b'
        }
        results['checks'].append(check_length)
    
    # CHECK 5: Required Weld Size
    if weld_data.weld_type == 'fillet':
        omega = 2.00
        w_required = q_per_weld * omega / (0.6 * FEXX * 0.707)
        w_required = max(w_required, w_min)
        results['w_required'] = w_required
        check_required = {
            'name': 'Required vs Provided Weld Size',
            'demand': f'{w_required:.1f} mm required',
            'capacity': f'{weld_data.weld_size:.1f} mm provided',
            'ok': weld_data.weld_size >= w_required,
            'reference': 'Design calculation'
        }
        results['checks'].append(check_required)
    
    # Overall Status
    all_ok = all(check['ok'] for check in results['checks'])
    results['ok'] = all_ok
    ratios = [check.get('ratio', 0) for check in results['checks'] if 'ratio' in check]
    results['ratio'] = max(ratios) if ratios else (0 if all_ok else 999)
    results['status'] = '✓ OK' if all_ok else '✗ NG'
    
    return results


def check_transverse_stiffener(sec: Section, Fy: float, stiff: StiffenerData) -> Dict:
    results = {'ok': True, 'checks': [], 'required': False}
    
    if not stiff.has_transverse:
        return results
    
    a = stiff.trans_spacing
    t_st = stiff.trans_t
    b_st = stiff.trans_b
    hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
    
    if a <= 0 or hw <= 0:
        return results
    
    results['required'] = True
    a_h = a / hw
    
    # (1) Width-to-thickness ratio (G2-12)
    bt_limit = 0.56 * math.sqrt(E_STEEL / Fy)
    bt_ratio = b_st / t_st if t_st > 0 else 999
    bt_ok = bt_ratio <= bt_limit
    results['checks'].append({
        'name': 'b/t Ratio',
        'demand': f"{bt_ratio:.1f}",
        'capacity': f"≤ {bt_limit:.1f}",
        'ok': bt_ok
    })
    
    # (2) Minimum moment of inertia (G2-13)
    j = max(2.5 / (a_h**2) - 2, 0.5)
    Ist_req = b_st * sec.tw**3 * j
    
    # Provided I (single plate about web face)
    Ist_prov = (1/12) * t_st * b_st**3 + t_st * b_st * (b_st/2)**2
    I_ok = Ist_prov >= Ist_req
    
    results['checks'].append({
        'name': 'Moment of Inertia',
        'demand': f"{Ist_req/1e4:.1f} cm⁴",
        'capacity': f"{Ist_prov/1e4:.1f} cm⁴",
        'ok': I_ok
    })
    
    # (3) Minimum width (practical)
    b_min = hw / 30 + sec.tw
    b_ok = b_st >= b_min
    results['checks'].append({
        'name': 'Min Width',
        'demand': f"≥ {b_min:.0f} mm",
        'capacity': f"{b_st:.0f} mm",
        'ok': b_ok
    })
    
    results['ok'] = bt_ok and I_ok and b_ok
    results['Ist_req'] = Ist_req
    results['Ist_prov'] = Ist_prov
    
    return results


def check_bearing_stiffener(sec: Section, Fy: float, Pu: float, 
                            stiff: StiffenerData, at_support: bool = True) -> Dict:
    results = {'ok': True, 'checks': [], 'required': False}
    
    if not stiff.has_bearing:
        return results
    
    t_st = stiff.bearing_t
    b_st = stiff.bearing_b
    hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
    
    results['required'] = True
    
    # (1) Width-to-thickness ratio
    bt_limit = 0.56 * math.sqrt(E_STEEL / Fy)
    bt_ratio = b_st / t_st if t_st > 0 else 999
    bt_ok = bt_ratio <= bt_limit
    results['checks'].append({
        'name': 'b/t Ratio',
        'demand': f"{bt_ratio:.1f}",
        'capacity': f"≤ {bt_limit:.1f}",
        'ok': bt_ok
    })
    
    # (2) Column capacity check
    # Effective web width
    web_eff = 25 * sec.tw if at_support else 24 * sec.tw
    
    # Effective area (pair of stiffeners + web)
    A_st = 2 * b_st * t_st
    A_web = web_eff * sec.tw
    A_eff = A_st + A_web
    
    # Moment of inertia about web centerline
    I_st = 2 * ((1/12) * t_st * b_st**3 + t_st * b_st * (b_st/2 + sec.tw/2)**2)
    I_web = (1/12) * web_eff * sec.tw**3
    I_eff = I_st + I_web
    
    # Radius of gyration
    r = math.sqrt(I_eff / A_eff) if A_eff > 0 else 1
    
    # Effective length (0.75*hw for bearing stiffeners)
    K = 0.75
    L_eff = K * hw
    KL_r = L_eff / r if r > 0 else 999
    
    # Column capacity per AISC E3
    Fe = math.pi**2 * E_STEEL / KL_r**2 if KL_r > 0 else Fy
    if KL_r <= 4.71 * math.sqrt(E_STEEL / Fy):
        Fcr = Fy * (0.658 ** (Fy / Fe))
    else:
        Fcr = 0.877 * Fe
    
    Pn = Fcr * A_eff / 1000  # kN
    Pa = Pn / OMEGA_COMP
    col_ok = Pu <= Pa
    
    results['checks'].append({
        'name': 'Column Capacity',
        'demand': f"{Pu:.1f} kN",
        'capacity': f"{Pa:.1f} kN",
        'ok': col_ok
    })
    
    # (3) Bearing check
    clip = 25  # mm typical clip at web-flange junction
    A_bearing = 2 * (b_st - clip) * t_st
    Pb = 1.8 * Fy * A_bearing / 1000 / OMEGA_BEARING
    bear_ok = Pu <= Pb
    
    results['checks'].append({
        'name': 'Bearing',
        'demand': f"{Pu:.1f} kN",
        'capacity': f"{Pb:.1f} kN",
        'ok': bear_ok
    })
    
    results['ok'] = bt_ok and col_ok and bear_ok
    results['Pn'] = Pn
    results['Pa'] = Pa
    results['ratio'] = Pu / Pa if Pa > 0 else 999
    
    return results


def check_longitudinal_stiffener(sec: Section, Fy: float, stiff: StiffenerData) -> Dict:
    results = {'ok': True, 'checks': [], 'required': False}
    
    if not stiff.has_longitudinal:
        return results
    
    t_st = stiff.long_t
    b_st = stiff.long_b
    hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
    
    results['required'] = True
    
    # (1) Width-to-thickness ratio
    bt_limit = 0.56 * math.sqrt(E_STEEL / Fy)
    bt_ratio = b_st / t_st if t_st > 0 else 999
    bt_ok = bt_ratio <= bt_limit
    results['checks'].append({
        'name': 'b/t Ratio',
        'demand': f"{bt_ratio:.1f}",
        'capacity': f"≤ {bt_limit:.1f}",
        'ok': bt_ok
    })
    
    # (2) Minimum moment of inertia
    Icr = hw * sec.tw**3 * 2.4
    Il_prov = (1/12) * t_st * b_st**3
    I_ok = Il_prov >= Icr * 0.5  # Allow 50% for practical purposes
    
    results['checks'].append({
        'name': 'Moment of Inertia',
        'demand': f"≥ {Icr*0.5/1e4:.1f} cm⁴",
        'capacity': f"{Il_prov/1e4:.1f} cm⁴",
        'ok': I_ok
    })
    
    # (3) Position check
    pos_ok = 0.15 <= stiff.long_position <= 0.35
    results['checks'].append({
        'name': 'Position',
        'demand': f"0.15-0.35 h",
        'capacity': f"{stiff.long_position:.2f} h",
        'ok': pos_ok
    })
    
    results['ok'] = bt_ok and I_ok and pos_ok
    results['Icr'] = Icr
    results['Il_prov'] = Il_prov
    
    return results


def design_bearing_stiffener(sec: Section, Fy: float, Pu: float,
                             at_support: bool = True) -> Dict:
    hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
    
    # Start with minimum practical dimensions
    t_st = 10  # mm
    b_st = max(hw / 30 + sec.tw, 50)  # mm
    
    # Check b/t limit
    bt_limit = 0.56 * math.sqrt(E_STEEL / Fy)
    
    # Iterate to find adequate size
    for _ in range(20):
        # Check b/t
        if b_st / t_st > bt_limit:
            t_st = math.ceil(b_st / bt_limit)
        
        # Calculate capacity
        web_eff = 25 * sec.tw if at_support else 24 * sec.tw
        A_st = 2 * b_st * t_st
        A_web = web_eff * sec.tw
        A_eff = A_st + A_web
        
        I_st = 2 * ((1/12) * t_st * b_st**3 + t_st * b_st * (b_st/2 + sec.tw/2)**2)
        I_web = (1/12) * web_eff * sec.tw**3
        I_eff = I_st + I_web
        
        r = math.sqrt(I_eff / A_eff)
        K = 0.75
        L_eff = K * hw
        KL_r = L_eff / r
        
        Fe = math.pi**2 * E_STEEL / KL_r**2
        if KL_r <= 4.71 * math.sqrt(E_STEEL / Fy):
            Fcr = Fy * (0.658 ** (Fy / Fe))
        else:
            Fcr = 0.877 * Fe
        
        Pa = Fcr * A_eff / 1000 / OMEGA_COMP
        
        if Pa >= Pu:
            break
        
        # Increase size
        b_st += 10
        t_st = max(t_st, math.ceil(b_st / bt_limit))
    
    return {
        'b_st': round(b_st, 0),
        't_st': round(t_st, 0),
        'Pa': Pa,
        'ratio': Pu / Pa if Pa > 0 else 999
    }



import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_influence_diagrams(results: CriticalResults, beam_span: float) -> go.Figure:
    positions = [r.step_position for r in results.all_results]
    R_A = [r.R_A for r in results.all_results]
    R_B = [r.R_B for r in results.all_results]
    M_max = [r.M_max for r in results.all_results]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Moment Envelope', 'Left Reaction (R_A)', 'Right Reaction (R_B)'),
        vertical_spacing=0.1
    )
    
    # Moment
    fig.add_trace(go.Scatter(
        x=positions, y=M_max, mode='lines+markers', name='M_max',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy', fillcolor='rgba(46,134,171,0.3)'
    ), row=1, col=1)
    
    idx_m = M_max.index(max(M_max))
    fig.add_trace(go.Scatter(
        x=[positions[idx_m]], y=[M_max[idx_m]], mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name=f'M_max = {results.M_max:.0f} kN-m'
    ), row=1, col=1)
    
    # R_A
    fig.add_trace(go.Scatter(
        x=positions, y=R_A, mode='lines+markers', name='R_A',
        line=dict(color='#A23B72', width=2),
        fill='tozeroy', fillcolor='rgba(162,59,114,0.3)'
    ), row=2, col=1)
    
    idx_ra = R_A.index(max(R_A))
    fig.add_trace(go.Scatter(
        x=[positions[idx_ra]], y=[R_A[idx_ra]], mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name=f'R_A_max = {results.R_A_max:.0f} kN'
    ), row=2, col=1)
    
    # R_B
    fig.add_trace(go.Scatter(
        x=positions, y=R_B, mode='lines+markers', name='R_B',
        line=dict(color='#F18F01', width=2),
        fill='tozeroy', fillcolor='rgba(241,143,1,0.3)'
    ), row=3, col=1)
    
    idx_rb = R_B.index(max(R_B))
    fig.add_trace(go.Scatter(
        x=[positions[idx_rb]], y=[R_B[idx_rb]], mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name=f'R_B_max = {results.R_B_max:.0f} kN'
    ), row=3, col=1)
    
    fig.update_xaxes(title_text="First Wheel Position (m)", row=3, col=1)
    fig.update_yaxes(title_text="Moment (kN-m)", row=1, col=1)
    fig.update_yaxes(title_text="Reaction (kN)", row=2, col=1)
    fig.update_yaxes(title_text="Reaction (kN)", row=3, col=1)
    
    fig.update_layout(
        height=650,
        showlegend=False,
        title_text="Moving Load Analysis - Influence Diagrams"
    )
    
    return fig


def plot_section(sec: Section, stiff: Optional[StiffenerData] = None) -> go.Figure:
    fig = go.Figure()
    
    d = sec.d
    bf_top = sec.bf_top if sec.bf_top > 0 else sec.bf
    bf_bot = sec.bf_bot if sec.bf_bot > 0 else sec.bf
    tf_top = sec.tf_top if sec.tf_top > 0 else sec.tf
    tf_bot = sec.tf_bot if sec.tf_bot > 0 else sec.tf
    tw = sec.tw
    hw = sec.hw if sec.hw > 0 else d - tf_top - tf_bot
    
    # Top flange
    fig.add_trace(go.Scatter(
        x=[-bf_top/2, bf_top/2, bf_top/2, -bf_top/2, -bf_top/2],
        y=[d-tf_top, d-tf_top, d, d, d-tf_top],
        mode='lines', fill='toself', fillcolor='#3498DB',
        line=dict(color='#2C3E50', width=2), name='Top Flange'
    ))
    
    # Web
    fig.add_trace(go.Scatter(
        x=[-tw/2, tw/2, tw/2, -tw/2, -tw/2],
        y=[tf_bot, tf_bot, d-tf_top, d-tf_top, tf_bot],
        mode='lines', fill='toself', fillcolor='#3498DB',
        line=dict(color='#2C3E50', width=2), name='Web'
    ))
    
    # Bottom flange
    fig.add_trace(go.Scatter(
        x=[-bf_bot/2, bf_bot/2, bf_bot/2, -bf_bot/2, -bf_bot/2],
        y=[0, 0, tf_bot, tf_bot, 0],
        mode='lines', fill='toself', fillcolor='#3498DB',
        line=dict(color='#2C3E50', width=2), name='Bottom Flange'
    ))
    
    # Cap channel - INVERTED (flanges pointing DOWN onto beam top flange)
    if sec.has_cap and sec.cap_d > 0:
        # Get actual channel dimensions from database if available
        cap_bf = sec.cap_d * 0.35  # Channel flange width (approximate)
        cap_tf = sec.cap_d * 0.12  # Channel flange thickness
        cap_tw = sec.cap_d * 0.08  # Channel web thickness
        
        # The channel sits INVERTED on top of the beam:
        # - Web is horizontal at the top
        # - Flanges hang down on each side
        
        cap_bottom = d  # Bottom of channel = top of beam
        cap_top = d + cap_tw  # Top of channel web
        flange_bottom = d - (sec.cap_d - cap_tw)  # Bottom of hanging flanges
        
        # But flanges can't go below beam top flange, so just show them going down a bit
        flange_depth = min(sec.cap_d - cap_tw, tf_top * 0.8)  # Flanges hang down
        
        # Horizontal web at top
        fig.add_trace(go.Scatter(
            x=[-cap_bf, cap_bf, cap_bf, -cap_bf, -cap_bf],
            y=[d + cap_tw, d + cap_tw, d, d, d + cap_tw],
            mode='lines', fill='toself', fillcolor='#E74C3C',
            line=dict(color='#C0392B', width=1.5), name='Cap Channel'
        ))
        
        # Left flange (hanging down)
        fig.add_trace(go.Scatter(
            x=[-cap_bf, -cap_bf + cap_tf, -cap_bf + cap_tf, -cap_bf, -cap_bf],
            y=[d, d, d - flange_depth, d - flange_depth, d],
            mode='lines', fill='toself', fillcolor='#E74C3C',
            line=dict(color='#C0392B', width=1.5), showlegend=False
        ))
        
        # Right flange (hanging down)
        fig.add_trace(go.Scatter(
            x=[cap_bf - cap_tf, cap_bf, cap_bf, cap_bf - cap_tf, cap_bf - cap_tf],
            y=[d, d, d - flange_depth, d - flange_depth, d],
            mode='lines', fill='toself', fillcolor='#E74C3C',
            line=dict(color='#C0392B', width=1.5), showlegend=False
        ))
    
    # Stiffeners
    if stiff:
        stiff_color = '#27AE60'
        
        if stiff.has_bearing:
            b = stiff.bearing_b
            t = stiff.bearing_t
            # Right stiffener
            fig.add_trace(go.Scatter(
                x=[tw/2, tw/2 + b, tw/2 + b, tw/2],
                y=[tf_bot, tf_bot, d - tf_top, d - tf_top],
                mode='lines', fill='toself', fillcolor=stiff_color,
                line=dict(color='#1E8449', width=1), name='Bearing Stiff'
            ))
            # Left stiffener
            fig.add_trace(go.Scatter(
                x=[-tw/2, -tw/2 - b, -tw/2 - b, -tw/2],
                y=[tf_bot, tf_bot, d - tf_top, d - tf_top],
                mode='lines', fill='toself', fillcolor=stiff_color,
                line=dict(color='#1E8449', width=1), showlegend=False
            ))
        
        if stiff.has_longitudinal:
            b = stiff.long_b
            t = stiff.long_t
            y_pos = d - tf_top - stiff.long_position * hw
            fig.add_trace(go.Scatter(
                x=[tw/2, tw/2 + b, tw/2 + b, tw/2],
                y=[y_pos - t/2, y_pos - t/2, y_pos + t/2, y_pos + t/2],
                mode='lines', fill='toself', fillcolor='#9B59B6',
                line=dict(color='#7D3C98', width=1), name='Long. Stiff'
            ))
    
    # Dimensions
    max_bf = max(bf_top, bf_bot)
    fig.add_annotation(x=max_bf/2 + 30, y=d/2, text=f'd={d:.0f}',
                      showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0, y=-25, text=f'bf={max_bf:.0f}',
                      showarrow=False, font=dict(size=10))
    fig.add_annotation(x=tw/2 + 20, y=d/2, text=f'tw={tw:.1f}',
                      showarrow=False, font=dict(size=9))
    
    title = f'Section: {sec.name}'
    if sec.has_cap:
        title += f' + {sec.cap_name}'
    
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        height=450,
        margin=dict(l=20, r=20, t=40, b=40)
    )
    
    return fig


def plot_beam_diagram(beam_span: float, wheel_positions: list, 
                      wheel_loads: list, R_A: float, R_B: float) -> go.Figure:
    fig = go.Figure()
    
    # Beam line
    fig.add_trace(go.Scatter(
        x=[0, beam_span], y=[0, 0],
        mode='lines', line=dict(color='#2C3E50', width=8),
        name='Beam'
    ))
    
    # Supports
    support_size = beam_span * 0.03
    # Left support (pin)
    fig.add_trace(go.Scatter(
        x=[0, -support_size, support_size, 0],
        y=[0, -support_size*1.5, -support_size*1.5, 0],
        mode='lines', fill='toself', fillcolor='#7F8C8D',
        line=dict(color='#2C3E50', width=2), name='Support'
    ))
    # Right support (roller)
    fig.add_trace(go.Scatter(
        x=[beam_span, beam_span - support_size, beam_span + support_size, beam_span],
        y=[0, -support_size*1.5, -support_size*1.5, 0],
        mode='lines', fill='toself', fillcolor='#7F8C8D',
        line=dict(color='#2C3E50', width=2), showlegend=False
    ))
    
    # Wheel loads (arrows pointing down)
    max_load = max(wheel_loads) if wheel_loads else 1
    arrow_scale = beam_span * 0.15 / max_load
    
    for pos, load in zip(wheel_positions, wheel_loads):
        arrow_len = load * arrow_scale
        fig.add_annotation(
            x=pos, y=0, ax=pos, ay=arrow_len,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=2, arrowcolor='#E74C3C'
        )
        fig.add_annotation(
            x=pos, y=arrow_len + beam_span*0.02,
            text=f'{load:.0f} kN', showarrow=False,
            font=dict(size=10, color='#E74C3C')
        )
    
    # Reactions (arrows pointing up)
    r_scale = beam_span * 0.1 / max(R_A, R_B)
    
    fig.add_annotation(
        x=0, y=0, ax=0, ay=-R_A * r_scale - support_size*2,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowwidth=2, arrowcolor='#27AE60'
    )
    fig.add_annotation(
        x=0, y=-R_A * r_scale - support_size*2 - beam_span*0.03,
        text=f'R_A={R_A:.0f} kN', showarrow=False,
        font=dict(size=10, color='#27AE60')
    )
    
    fig.add_annotation(
        x=beam_span, y=0, ax=beam_span, ay=-R_B * r_scale - support_size*2,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowwidth=2, arrowcolor='#27AE60'
    )
    fig.add_annotation(
        x=beam_span, y=-R_B * r_scale - support_size*2 - beam_span*0.03,
        text=f'R_B={R_B:.0f} kN', showarrow=False,
        font=dict(size=10, color='#27AE60')
    )
    
    fig.update_layout(
        title='Beam Loading Diagram',
        xaxis=dict(title='Position (m)', showgrid=True),
        yaxis=dict(scaleanchor='x', showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=300
    )
    
    return fig


# ============================================================================
# PDF REPORT GENERATOR - ACADEMIC STYLE
# ============================================================================

def create_pdf_styles():
    """Create custom paragraph styles for academic report"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'], fontSize=22,
        spaceAfter=20, alignment=TA_CENTER, textColor=colors.HexColor('#1a365d'), fontName='Helvetica-Bold'))
    
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading1'], fontSize=13,
        spaceBefore=18, spaceAfter=10, textColor=colors.HexColor('#1a365d'), fontName='Helvetica-Bold',
        backColor=colors.HexColor('#e6f0ff'), borderPadding=5))
    
    styles.add(ParagraphStyle(name='SubsectionHeader', parent=styles['Heading2'], fontSize=11,
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#2c5282'), fontName='Helvetica-Bold'))
    
    styles.add(ParagraphStyle(name='BodyTextCustom', parent=styles['Normal'], fontSize=10,
        spaceBefore=3, spaceAfter=3, alignment=TA_JUSTIFY, leading=13))
    
    styles.add(ParagraphStyle(name='Equation', parent=styles['Normal'], fontSize=10,
        spaceBefore=6, spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#2d3748'), backColor=colors.HexColor('#f7fafc'), borderPadding=6))
    
    styles.add(ParagraphStyle(name='Reference', parent=styles['Normal'], fontSize=8,
        spaceBefore=2, spaceAfter=2, textColor=colors.HexColor('#4a5568'), fontName='Helvetica-Oblique'))
    
    styles.add(ParagraphStyle(name='ResultPass', parent=styles['Normal'], fontSize=10,
        spaceBefore=4, spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#22543d'), backColor=colors.HexColor('#c6f6d5'), borderPadding=6))
    
    styles.add(ParagraphStyle(name='ResultFail', parent=styles['Normal'], fontSize=10,
        spaceBefore=4, spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#742a2a'), backColor=colors.HexColor('#fed7d7'), borderPadding=6))
    
    return styles


def create_pdf_table(data, col_widths=None, header_color='#3182ce'):
    """Create a styled PDF table"""
    if col_widths is None:
        col_widths = [100] * len(data[0])
    
    table = Table(data, colWidths=col_widths)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4a5568')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
    ])
    table.setStyle(style)
    return table


def generate_academic_pdf_report(
    beam_span, Lb, steel_grade, Fy, Fu, sec, cranes, crane_class, 
    N_cycles, fatigue_cat, results, M_self, V_self, M_design, V_design,
    cmp, flex, shear, wly, wcr, fatigue, delta, delta_limit,
    stiff, trans_check, bearing_check, flex_ratio, shear_ratio, defl_ratio,
    check_fatigue_enabled=True, weld_check=None, project_name="Crane Runway Beam Design",
    project_number="", designer="", checker=""
):
    """Generate comprehensive academic-style PDF report"""
    
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                           topMargin=20*mm, bottomMargin=20*mm)
    
    styles = create_pdf_styles()
    story = []
    
    # Determine overall status
    overall_pass = all([
        flex_ratio <= 1.0, shear_ratio <= 1.0, 
        wly['ratio'] <= 1.0, wcr['ratio'] <= 1.0, defl_ratio <= 1.0,
        (fatigue['ratio'] <= 1.0 if check_fatigue_enabled else True)
    ])
    
    # ===================== TITLE PAGE =====================
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("CRANE RUNWAY BEAM", styles['ReportTitle']))
    story.append(Paragraph("STRUCTURAL DESIGN REPORT", styles['ReportTitle']))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "Per AISC 360-16 Specification (ASD Method)<br/>"
        "AISC Design Guide 7: Industrial Buildings<br/>"
        "CMAA 70/74 Crane Specifications",
        styles['BodyTextCustom']
    ))
    story.append(Spacer(1, 15*mm))
    
    # Project info table
    info_data = [
        ['PROJECT INFORMATION', ''],
        ['Project Name:', project_name or 'Crane Runway Design'],
        ['Project Number:', project_number or '-'],
        ['Designer:', designer or '-'],
        ['Checker:', checker or '-'],
        ['Date:', datetime.now().strftime('%B %d, %Y')],
        ['Software:', 'CraneRunwayPro V6.0'],
    ]
    story.append(create_pdf_table(info_data, col_widths=[55*mm, 75*mm], header_color='#2c5282'))
    story.append(Spacer(1, 15*mm))
    
    if overall_pass:
        story.append(Paragraph("<b>DESIGN STATUS: ✓ ALL CHECKS PASSED</b>", styles['ResultPass']))
    else:
        story.append(Paragraph("<b>DESIGN STATUS: ✗ REVISIONS REQUIRED</b>", styles['ResultFail']))
    
    story.append(PageBreak())
    
    # ===================== 1. INTRODUCTION =====================
    story.append(Paragraph("1. INTRODUCTION", styles['SectionHeader']))
    story.append(Paragraph(
        f"This report presents the structural design of a crane runway beam per AISC 360-16 "
        f"using the Allowable Stress Design (ASD) method. The beam spans {beam_span:.2f} m and "
        f"supports {len(cranes)} crane(s) classified as CMAA Class {crane_class}. The design "
        f"follows recommendations from AISC Design Guide 7: Industrial Buildings.",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph("<i>Reference: AISC 360-16, Specification for Structural Steel Buildings</i>", styles['Reference']))
    
    # ===================== 2. DESIGN DATA =====================
    story.append(Paragraph("2. DESIGN DATA", styles['SectionHeader']))
    
    story.append(Paragraph("2.1 Geometry", styles['SubsectionHeader']))
    geom_data = [['Parameter', 'Symbol', 'Value', 'Unit'],
                 ['Beam Span', 'L', f'{beam_span:.3f}', 'm'],
                 ['Unbraced Length', 'Lb', f'{Lb:.3f}', 'm']]
    story.append(create_pdf_table(geom_data, col_widths=[45*mm, 25*mm, 35*mm, 25*mm]))
    
    story.append(Paragraph("2.2 Material Properties", styles['SubsectionHeader']))
    mat_data = [['Property', 'Symbol', 'Value', 'Unit'],
                ['Steel Grade', '-', steel_grade, '-'],
                ['Yield Strength', 'Fy', f'{Fy}', 'MPa'],
                ['Ultimate Strength', 'Fu', f'{Fu}', 'MPa'],
                ['Elastic Modulus', 'E', '200,000', 'MPa']]
    story.append(create_pdf_table(mat_data, col_widths=[45*mm, 25*mm, 35*mm, 25*mm]))
    
    story.append(Paragraph("2.3 Section Properties", styles['SubsectionHeader']))
    sec_name = f"{sec.name}" + (f" + {sec.cap_name}" if sec.has_cap else "")
    tf_d = sec.tf if sec.tf > 0 else sec.tf_top
    bf_d = sec.bf if sec.bf > 0 else max(sec.bf_top, sec.bf_bot)
    sec_data = [['Property', 'Symbol', 'Value', 'Unit'],
                ['Section', '-', sec_name, '-'],
                ['Total Depth', 'd', f'{sec.d:.0f}', 'mm'],
                ['Flange Width', 'bf', f'{bf_d:.0f}', 'mm'],
                ['Flange Thickness', 'tf', f'{tf_d:.1f}', 'mm'],
                ['Web Thickness', 'tw', f'{sec.tw:.1f}', 'mm'],
                ['Moment of Inertia', 'Ix', f'{sec.Ix/1e6:.2f}×10⁶', 'mm⁴'],
                ['Section Modulus', 'Sx', f'{sec.Sx/1e3:.1f}×10³', 'mm³'],
                ['Plastic Modulus', 'Zx', f'{sec.Zx/1e3:.1f}×10³', 'mm³'],
                ['Unit Mass', 'w', f'{sec.mass:.1f}', 'kg/m']]
    story.append(create_pdf_table(sec_data, col_widths=[45*mm, 25*mm, 40*mm, 25*mm]))
    
    story.append(Paragraph("2.4 Crane Data", styles['SubsectionHeader']))
    crane_data = [['Crane', 'Capacity', 'Wheels', 'Wheel Base', 'Wmax×φ']]
    for i, c in enumerate(cranes):
        P_max, _ = c.calc_wheel_loads()
        crane_data.append([f'Crane {i+1}', f'{c.capacity_tonnes:.1f} T', f'{c.num_wheels}',
                          f'{c.get_total_wheel_base():.2f} m', f'{c.get_wheel_load_with_impact():.1f} kN'])
    story.append(create_pdf_table(crane_data, col_widths=[25*mm, 28*mm, 22*mm, 30*mm, 30*mm]))
    
    # Show wheel configuration for 4-wheel cranes
    for i, c in enumerate(cranes):
        if c.num_wheels == 4:
            wheel_pos = c.get_wheel_positions_relative()
            story.append(Paragraph(
                f"<b>Crane {i+1} - 4-Wheel Configuration:</b><br/>"
                f"W1(0) → [{c.wheel_spacing_12:.3f}m] → W2({wheel_pos[1]:.3f}m) → "
                f"[{c.wheel_spacing_23:.3f}m] → W3({wheel_pos[2]:.3f}m) → "
                f"[{c.wheel_spacing_34:.3f}m] → W4({wheel_pos[3]:.3f}m)",
                styles['BodyTextCustom']
            ))
    
    story.append(PageBreak())
    
    # ===================== 3. LOAD ANALYSIS =====================
    story.append(Paragraph("3. LOAD ANALYSIS", styles['SectionHeader']))
    
    story.append(Paragraph("3.1 Self-Weight", styles['SubsectionHeader']))
    w_self = sec.mass * 9.81 / 1000
    R_self = w_self * beam_span / 2
    story.append(Paragraph(f"Self-weight: w = m × g = {sec.mass:.1f} × 9.81 / 1000 = <b>{w_self:.3f} kN/m</b>", styles['Equation']))
    story.append(Paragraph(f"M_self = wL²/8 = {w_self:.3f} × {beam_span:.2f}² / 8 = <b>{M_self:.2f} kN-m</b>", styles['Equation']))
    
    story.append(PageBreak())
    
    # ===================== 3.2 MOVING LOAD ANALYSIS - DETAILED =====================
    story.append(Paragraph("3.2 Moving Load Analysis - Theory", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The moving load analysis determines the critical wheel positions that produce maximum "
        "internal forces and reactions. For crane runway beams, the analysis must consider multiple "
        "cranes operating simultaneously with minimum clearances between them.",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph("<i>Reference: AISC Design Guide 7, Chapter 3 - Crane Loads</i>", styles['Reference']))
    
    # Influence Line Theory
    story.append(Paragraph("3.2.1 Influence Line Method", styles['SubsectionHeader']))
    story.append(Paragraph(
        "Influence lines are used to determine the effect of moving loads at any position along the beam. "
        "For a simply supported beam of span L:",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        "<b>Reaction at A:</b> η_RA(x) = (L - x) / L",
        styles['Equation']
    ))
    story.append(Paragraph(
        "<b>Reaction at B:</b> η_RB(x) = x / L",
        styles['Equation']
    ))
    story.append(Paragraph(
        "<b>Moment at point a:</b> η_M(x) = x(L-a)/L for x ≤ a; η_M(x) = a(L-x)/L for x > a",
        styles['Equation']
    ))
    
    # Wheel Configuration
    story.append(Paragraph("3.2.2 Wheel Configuration", styles['SubsectionHeader']))
    
    # Create wheel configuration table
    wheel_config = [['Crane', 'Wheels', 'Wheel Base (m)', 'Buffer Left (m)', 'Buffer Right (m)', 'Total Length (m)']]
    total_wheels = 0
    for i, c in enumerate(cranes):
        total_len = c.buffer_left + c.wheel_base + c.buffer_right
        wheel_config.append([
            f'Crane {i+1}', f'{c.num_wheels}', f'{c.wheel_base:.3f}',
            f'{c.buffer_left:.3f}', f'{c.buffer_right:.3f}', f'{total_len:.3f}'
        ])
        total_wheels += c.num_wheels
    story.append(create_pdf_table(wheel_config, col_widths=[22*mm, 18*mm, 28*mm, 28*mm, 28*mm, 28*mm]))
    story.append(Spacer(1, 3*mm))
    
    # Minimum crane gap explanation
    if len(cranes) > 1:
        min_gap = cranes[0].buffer_right + cranes[1].buffer_left
        story.append(Paragraph(
            f"<b>Minimum Crane Gap:</b> The minimum distance between adjacent cranes is determined by "
            f"the buffer distances. For Crane 1 and Crane 2:",
            styles['BodyTextCustom']
        ))
        story.append(Paragraph(
            f"Gap_min = Buffer_right(Crane 1) + Buffer_left(Crane 2) = {cranes[0].buffer_right:.3f} + {cranes[1].buffer_left:.3f} = <b>{min_gap:.3f} m</b>",
            styles['Equation']
        ))
    
    # Analysis Parameters
    story.append(Paragraph("3.2.3 Analysis Parameters", styles['SubsectionHeader']))
    story.append(Paragraph(
        f"• Beam Span: L = {beam_span:.3f} m<br/>"
        f"• Total number of wheels: {total_wheels}<br/>"
        f"• Number of load positions analyzed: {len(results.all_results)}<br/>"
        f"• Step size: {beam_span / max(len(results.all_results)-1, 1) * 1000:.1f} mm",
        styles['BodyTextCustom']
    ))
    
    story.append(PageBreak())
    
    # ===================== 3.3 CRITICAL LOAD POSITIONS =====================
    story.append(Paragraph("3.3 Critical Wheel Positions", styles['SectionHeader']))
    
    # Maximum Moment Position
    story.append(Paragraph("3.3.1 Position for Maximum Bending Moment", styles['SubsectionHeader']))
    story.append(Paragraph(
        f"The maximum bending moment occurs at <b>x = {results.M_max_location:.3f} m</b> from the left support.",
        styles['BodyTextCustom']
    ))
    
    # Create moment diagram description
    story.append(Paragraph("<b>Wheel Positions at Maximum Moment:</b>", styles['BodyTextCustom']))
    
    wheel_pos_data = [['Wheel No.', 'Position (m)', 'Load (kN)', 'From Left Support']]
    wheel_loads = []
    for c in cranes:
        P = c.get_wheel_load_with_impact()
        wheel_loads.extend([P] * c.num_wheels)
    
    for i, pos in enumerate(results.M_max_wheel_positions):
        if i < len(wheel_loads):
            wheel_pos_data.append([
                f'W{i+1}', f'{pos:.3f}', f'{wheel_loads[i]:.2f}',
                f'{pos/beam_span*100:.1f}% of span'
            ])
    story.append(create_pdf_table(wheel_pos_data, col_widths=[25*mm, 35*mm, 30*mm, 45*mm]))
    story.append(Spacer(1, 3*mm))
    
    # Draw beam diagram for max moment using ReportLab Drawing
    story.append(Paragraph("<b>Figure 3.1: Wheel Arrangement for Maximum Moment</b>", styles['BodyTextCustom']))
    
    # Create beam diagram
    drawing_width = 170*mm
    drawing_height = 60*mm
    d = Drawing(drawing_width, drawing_height)
    
    # Scale factors
    beam_draw_length = 150*mm
    x_offset = 10*mm
    y_beam = 25*mm
    scale = beam_draw_length / beam_span
    
    # Draw beam
    d.add(Line(x_offset, y_beam, x_offset + beam_draw_length, y_beam, strokeWidth=2, strokeColor=colors.HexColor('#2c5282')))
    
    # Draw supports (triangles)
    # Left support
    d.add(Line(x_offset, y_beam, x_offset - 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d.add(Line(x_offset, y_beam, x_offset + 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d.add(Line(x_offset - 4*mm, y_beam - 8*mm, x_offset + 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d.add(String(x_offset, y_beam - 15*mm, 'A', fontSize=8, textAnchor='middle'))
    
    # Right support
    d.add(Line(x_offset + beam_draw_length, y_beam, x_offset + beam_draw_length - 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d.add(Line(x_offset + beam_draw_length, y_beam, x_offset + beam_draw_length + 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d.add(Line(x_offset + beam_draw_length - 4*mm, y_beam - 8*mm, x_offset + beam_draw_length + 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d.add(String(x_offset + beam_draw_length, y_beam - 15*mm, 'B', fontSize=8, textAnchor='middle'))
    
    # Draw wheels as arrows
    arrow_length = 15*mm
    for i, pos in enumerate(results.M_max_wheel_positions):
        if pos >= 0 and pos <= beam_span:
            x_pos = x_offset + pos * scale
            # Arrow line
            d.add(Line(x_pos, y_beam + arrow_length, x_pos, y_beam + 2*mm, strokeWidth=1.5, strokeColor=colors.HexColor('#e53e3e')))
            # Arrow head
            d.add(Line(x_pos, y_beam + 2*mm, x_pos - 2*mm, y_beam + 6*mm, strokeWidth=1.5, strokeColor=colors.HexColor('#e53e3e')))
            d.add(Line(x_pos, y_beam + 2*mm, x_pos + 2*mm, y_beam + 6*mm, strokeWidth=1.5, strokeColor=colors.HexColor('#e53e3e')))
            # Wheel label
            d.add(String(x_pos, y_beam + arrow_length + 3*mm, f'W{i+1}', fontSize=7, textAnchor='middle'))
    
    # Draw moment location marker
    x_mmax = x_offset + results.M_max_location * scale
    d.add(Line(x_mmax, y_beam - 2*mm, x_mmax, y_beam - 10*mm, strokeWidth=1, strokeColor=colors.HexColor('#38a169'), strokeDashArray=[2,2]))
    d.add(String(x_mmax, y_beam - 18*mm, f'Mmax', fontSize=7, textAnchor='middle', fillColor=colors.HexColor('#38a169')))
    
    # Dimension line for span
    d.add(Line(x_offset, y_beam - 22*mm, x_offset + beam_draw_length, y_beam - 22*mm, strokeWidth=0.5))
    d.add(Line(x_offset, y_beam - 20*mm, x_offset, y_beam - 24*mm, strokeWidth=0.5))
    d.add(Line(x_offset + beam_draw_length, y_beam - 20*mm, x_offset + beam_draw_length, y_beam - 24*mm, strokeWidth=0.5))
    d.add(String(x_offset + beam_draw_length/2, y_beam - 28*mm, f'L = {beam_span:.2f} m', fontSize=8, textAnchor='middle'))
    
    story.append(d)
    story.append(Spacer(1, 5*mm))
    
    # Moment calculation
    story.append(Paragraph("<b>Maximum Moment Calculation:</b>", styles['BodyTextCustom']))
    story.append(Paragraph(
        f"Using the influence line method, the maximum moment from crane loads is calculated by "
        f"summing the products of each wheel load and its influence ordinate at the critical section:",
        styles['BodyTextCustom']
    ))
    story.append(Paragraph(
        f"M_crane = Σ(P_i × η_i) = <b>{results.M_max:.2f} kN-m</b> at x = {results.M_max_location:.3f} m",
        styles['Equation']
    ))
    
    story.append(Spacer(1, 5*mm))
    
    # Maximum Reaction Position
    story.append(Paragraph("3.3.2 Position for Maximum Support Reaction", styles['SubsectionHeader']))
    story.append(Paragraph(
        f"The maximum reaction at support A occurs when the wheels are positioned to maximize "
        f"the influence line ordinates for R_A (wheels closest to support A).",
        styles['BodyTextCustom']
    ))
    
    # Reaction calculation explanation
    story.append(Paragraph("<b>Wheel Positions at Maximum Reaction R_A:</b>", styles['BodyTextCustom']))
    
    # For max reaction, first wheel should be close to support A
    react_pos_data = [['Wheel No.', 'Position (m)', 'Load (kN)', 'η_RA = (L-x)/L', 'Contribution (kN)']]
    for i, pos in enumerate(results.R_A_max_wheel_positions):
        if i < len(wheel_loads) and pos >= 0 and pos <= beam_span:
            eta = (beam_span - pos) / beam_span
            contrib = wheel_loads[i] * eta
            react_pos_data.append([
                f'W{i+1}', f'{pos:.3f}', f'{wheel_loads[i]:.2f}',
                f'{eta:.4f}', f'{contrib:.2f}'
            ])
    story.append(create_pdf_table(react_pos_data, col_widths=[22*mm, 28*mm, 25*mm, 35*mm, 32*mm]))
    story.append(Spacer(1, 3*mm))
    
    # Draw beam diagram for max reaction
    story.append(Paragraph("<b>Figure 3.2: Wheel Arrangement for Maximum Reaction at A</b>", styles['BodyTextCustom']))
    
    d2 = Drawing(drawing_width, drawing_height)
    
    # Draw beam
    d2.add(Line(x_offset, y_beam, x_offset + beam_draw_length, y_beam, strokeWidth=2, strokeColor=colors.HexColor('#2c5282')))
    
    # Left support (highlighted)
    d2.add(Line(x_offset, y_beam, x_offset - 4*mm, y_beam - 8*mm, strokeWidth=2, strokeColor=colors.HexColor('#e53e3e')))
    d2.add(Line(x_offset, y_beam, x_offset + 4*mm, y_beam - 8*mm, strokeWidth=2, strokeColor=colors.HexColor('#e53e3e')))
    d2.add(Line(x_offset - 4*mm, y_beam - 8*mm, x_offset + 4*mm, y_beam - 8*mm, strokeWidth=2, strokeColor=colors.HexColor('#e53e3e')))
    d2.add(String(x_offset, y_beam - 15*mm, 'A (max)', fontSize=8, textAnchor='middle', fillColor=colors.HexColor('#e53e3e')))
    
    # Right support
    d2.add(Line(x_offset + beam_draw_length, y_beam, x_offset + beam_draw_length - 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d2.add(Line(x_offset + beam_draw_length, y_beam, x_offset + beam_draw_length + 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d2.add(Line(x_offset + beam_draw_length - 4*mm, y_beam - 8*mm, x_offset + beam_draw_length + 4*mm, y_beam - 8*mm, strokeWidth=1.5))
    d2.add(String(x_offset + beam_draw_length, y_beam - 15*mm, 'B', fontSize=8, textAnchor='middle'))
    
    # Draw wheels
    for i, pos in enumerate(results.R_A_max_wheel_positions):
        if pos >= 0 and pos <= beam_span:
            x_pos = x_offset + pos * scale
            d2.add(Line(x_pos, y_beam + arrow_length, x_pos, y_beam + 2*mm, strokeWidth=1.5, strokeColor=colors.HexColor('#e53e3e')))
            d2.add(Line(x_pos, y_beam + 2*mm, x_pos - 2*mm, y_beam + 6*mm, strokeWidth=1.5, strokeColor=colors.HexColor('#e53e3e')))
            d2.add(Line(x_pos, y_beam + 2*mm, x_pos + 2*mm, y_beam + 6*mm, strokeWidth=1.5, strokeColor=colors.HexColor('#e53e3e')))
            d2.add(String(x_pos, y_beam + arrow_length + 3*mm, f'W{i+1}', fontSize=7, textAnchor='middle'))
    
    # Dimension line
    d2.add(Line(x_offset, y_beam - 22*mm, x_offset + beam_draw_length, y_beam - 22*mm, strokeWidth=0.5))
    d2.add(Line(x_offset, y_beam - 20*mm, x_offset, y_beam - 24*mm, strokeWidth=0.5))
    d2.add(Line(x_offset + beam_draw_length, y_beam - 20*mm, x_offset + beam_draw_length, y_beam - 24*mm, strokeWidth=0.5))
    d2.add(String(x_offset + beam_draw_length/2, y_beam - 28*mm, f'L = {beam_span:.2f} m', fontSize=8, textAnchor='middle'))
    
    # Reaction arrow at A
    d2.add(Line(x_offset, y_beam - 8*mm, x_offset, y_beam - 18*mm, strokeWidth=2, strokeColor=colors.HexColor('#38a169')))
    d2.add(Line(x_offset - 2*mm, y_beam - 14*mm, x_offset, y_beam - 18*mm, strokeWidth=2, strokeColor=colors.HexColor('#38a169')))
    d2.add(Line(x_offset + 2*mm, y_beam - 14*mm, x_offset, y_beam - 18*mm, strokeWidth=2, strokeColor=colors.HexColor('#38a169')))
    
    story.append(d2)
    story.append(Spacer(1, 3*mm))
    
    story.append(Paragraph(
        f"R_A,crane = Σ(P_i × η_RA,i) = <b>{results.R_A_max:.2f} kN</b>",
        styles['Equation']
    ))
    
    story.append(PageBreak())
    
    # ===================== 3.4 LOAD SUMMARY =====================
    story.append(Paragraph("3.4 Design Load Summary", styles['SubsectionHeader']))
    
    story.append(Paragraph("<b>Vertical Loads:</b>", styles['BodyTextCustom']))
    load_summary = [['Load Type', 'Moment (kN-m)', 'Shear (kN)', 'Reaction A (kN)', 'Reaction B (kN)'],
                    ['Self-weight', f'{M_self:.2f}', f'{V_self:.2f}', f'{R_self:.2f}', f'{R_self:.2f}'],
                    ['Crane (with impact)', f'{results.M_max:.2f}', f'{results.V_max:.2f}', f'{results.R_A_max:.2f}', f'{results.R_B_max:.2f}'],
                    ['TOTAL DESIGN', f'{M_design:.2f}', f'{V_design:.2f}', f'{results.R_A_max + R_self:.2f}', f'{results.R_B_max + R_self:.2f}']]
    story.append(create_pdf_table(load_summary, col_widths=[35*mm, 30*mm, 28*mm, 32*mm, 32*mm]))
    story.append(Spacer(1, 5*mm))
    
    # Lateral and longitudinal loads
    total_lat = sum(c.get_lateral_load_per_wheel() * c.num_wheels for c in cranes)
    max_long = max(c.get_longitudinal_force() for c in cranes)
    
    story.append(Paragraph("<b>Horizontal Loads:</b>", styles['BodyTextCustom']))
    horiz_loads = [['Load Type', 'Value', 'Application'],
                   ['Lateral Thrust (H)', f'{total_lat:.2f} kN', 'Top of rail, perpendicular to runway'],
                   ['Longitudinal Force (L)', f'{max_long:.2f} kN', 'Top of rail, along runway']]
    story.append(create_pdf_table(horiz_loads, col_widths=[45*mm, 35*mm, 70*mm]))
    
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(
        "<i>Note: Lateral thrust is calculated as 20% of the sum of lifted load and trolley weight. "
        "Longitudinal force is calculated as 10% of the maximum wheel loads per CMAA specifications.</i>",
        styles['Reference']
    ))
    
    story.append(PageBreak())
    
    # ===================== 4. SECTION CLASSIFICATION =====================
    story.append(Paragraph("4. SECTION CLASSIFICATION", styles['SectionHeader']))
    story.append(Paragraph("<i>Reference: AISC 360-16, Table B4.1b - Limiting Width-to-Thickness Ratios</i>", styles['Reference']))
    
    story.append(Paragraph("4.1 Flange Slenderness", styles['SubsectionHeader']))
    story.append(Paragraph(f"λf = bf/(2tf) = {bf_d:.0f}/(2×{tf_d:.1f}) = <b>{cmp['lambda_f']:.2f}</b>", styles['Equation']))
    story.append(Paragraph(f"Compact limit: λpf = 0.38√(E/Fy) = 0.38√(200000/{Fy}) = <b>{cmp['lambda_pf']:.2f}</b>", styles['Equation']))
    story.append(Paragraph(f"Noncompact limit: λrf = 1.0√(E/Fy) = <b>{cmp['lambda_rf']:.2f}</b>", styles['Equation']))
    if cmp['lambda_f'] <= cmp['lambda_pf']:
        story.append(Paragraph(f"Since λf = {cmp['lambda_f']:.2f} ≤ λpf = {cmp['lambda_pf']:.2f} → <b>Flange is COMPACT</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"<b>Flange Classification: {cmp['flange_class'].upper()}</b>", styles['BodyTextCustom']))
    
    story.append(Paragraph("4.2 Web Slenderness", styles['SubsectionHeader']))
    story.append(Paragraph(f"λw = h/tw = {sec.hw:.0f}/{sec.tw:.1f} = <b>{cmp['lambda_w']:.2f}</b>", styles['Equation']))
    story.append(Paragraph(f"Compact limit: λpw = 3.76√(E/Fy) = <b>{cmp['lambda_pw']:.2f}</b>", styles['Equation']))
    if cmp['lambda_w'] <= cmp['lambda_pw']:
        story.append(Paragraph(f"Since λw = {cmp['lambda_w']:.2f} ≤ λpw = {cmp['lambda_pw']:.2f} → <b>Web is COMPACT</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"<b>Web Classification: {cmp['web_class'].upper()}</b>", styles['BodyTextCustom']))
    
    story.append(PageBreak())
    
    # ===================== 5. FLEXURAL DESIGN =====================
    story.append(Paragraph("5. FLEXURAL DESIGN", styles['SectionHeader']))
    story.append(Paragraph("<i>Reference: AISC 360-16, Chapter F - Design of Members for Flexure</i>", styles['Reference']))
    
    story.append(Paragraph("5.1 Plastic Moment Capacity", styles['SubsectionHeader']))
    story.append(Paragraph(f"Plastic moment (Eq. F2-1): Mp = Fy × Zx = {Fy} × {sec.Zx/1e3:.1f}×10³ / 10⁶ = <b>{flex['Mp']:.2f} kN-m</b>", styles['Equation']))
    story.append(Paragraph(f"Yield moment: My = Fy × Sx = {Fy} × {sec.Sx/1e3:.1f}×10³ / 10⁶ = <b>{flex['My']:.2f} kN-m</b>", styles['Equation']))
    
    story.append(Paragraph("5.2 Lateral-Torsional Buckling Parameters", styles['SubsectionHeader']))
    story.append(Paragraph(f"Limiting length for yielding (Eq. F2-5): Lp = 1.76 × ry × √(E/Fy) = <b>{flex['Lp']/1000:.3f} m</b>", styles['Equation']))
    story.append(Paragraph(f"Limiting length for inelastic LTB (Eq. F2-6): Lr = <b>{flex['Lr']/1000:.3f} m</b>", styles['Equation']))
    story.append(Paragraph(f"Unbraced length: Lb = <b>{Lb:.3f} m</b>", styles['BodyTextCustom']))
    story.append(Paragraph(f"Moment gradient factor: Cb = <b>{flex['Cb']:.2f}</b>", styles['BodyTextCustom']))
    
    if Lb * 1000 <= flex['Lp']:
        story.append(Paragraph(f"Since Lb ≤ Lp → Yielding controls (Zone 1)", styles['BodyTextCustom']))
    elif Lb * 1000 <= flex['Lr']:
        story.append(Paragraph(f"Since Lp < Lb ≤ Lr → Inelastic LTB controls (Zone 2)", styles['BodyTextCustom']))
    else:
        story.append(Paragraph(f"Since Lb > Lr → Elastic LTB controls (Zone 3)", styles['BodyTextCustom']))
    
    story.append(Paragraph(f"<b>Governing Limit State:</b> {flex['limit_state']}", styles['BodyTextCustom']))
    
    story.append(Paragraph("5.3 Flexural Strength Check", styles['SubsectionHeader']))
    story.append(Paragraph(f"Nominal flexural strength: Mn = <b>{flex['Mn']:.2f} kN-m</b>", styles['Equation']))
    story.append(Paragraph(f"Allowable flexural strength (ASD): Mn/Ωb = {flex['Mn']:.2f} / 1.67 = <b>{flex['Mn_allow']:.2f} kN-m</b>", styles['Equation']))
    story.append(Paragraph(f"<b>Design Check: M_design / (Mn/Ωb) = {M_design:.2f} / {flex['Mn_allow']:.2f} = {flex_ratio:.3f}</b>", styles['Equation']))
    if flex_ratio <= 1.0:
        story.append(Paragraph(f"Ratio = {flex_ratio:.3f} ≤ 1.0 → <b>FLEXURE CHECK: PASS ✓</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"Ratio = {flex_ratio:.3f} > 1.0 → <b>FLEXURE CHECK: FAIL ✗</b>", styles['ResultFail']))
    
    # ===================== 6. SHEAR DESIGN =====================
    story.append(Paragraph("6. SHEAR DESIGN", styles['SectionHeader']))
    story.append(Paragraph("<i>Reference: AISC 360-16, Chapter G - Design of Members for Shear</i>", styles['Reference']))
    
    story.append(Paragraph(f"Web shear area: Aw = d × tw = {sec.d:.0f} × {sec.tw:.1f} = <b>{shear['Aw']:.0f} mm²</b>", styles['Equation']))
    story.append(Paragraph(f"Web slenderness: h/tw = <b>{shear['h_tw']:.2f}</b>", styles['BodyTextCustom']))
    story.append(Paragraph(f"Plate buckling coefficient: kv = <b>{shear['kv']:.2f}</b>", styles['BodyTextCustom']))
    story.append(Paragraph(f"Web shear coefficient: Cv1 = <b>{shear['Cv1']:.4f}</b>", styles['BodyTextCustom']))
    story.append(Paragraph(f"Nominal shear strength (Eq. G2-1): Vn = 0.6 × Fy × Aw × Cv1 = <b>{shear['Vn']:.2f} kN</b>", styles['Equation']))
    story.append(Paragraph(f"Allowable shear strength (ASD): Vn/Ωv = {shear['Vn']:.2f} / 1.50 = <b>{shear['Vn_allow']:.2f} kN</b>", styles['Equation']))
    story.append(Paragraph(f"<b>Design Check: V_design / (Vn/Ωv) = {V_design:.2f} / {shear['Vn_allow']:.2f} = {shear_ratio:.3f}</b>", styles['Equation']))
    if shear_ratio <= 1.0:
        story.append(Paragraph(f"<b>SHEAR CHECK: PASS ✓</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"<b>SHEAR CHECK: FAIL ✗</b>", styles['ResultFail']))
    
    story.append(PageBreak())
    
    # ===================== 7. WEB LOCAL EFFECTS =====================
    story.append(Paragraph("7. WEB LOCAL EFFECTS", styles['SectionHeader']))
    story.append(Paragraph("<i>Reference: AISC 360-16, Section J10 - Flanges and Webs with Concentrated Forces</i>", styles['Reference']))
    
    story.append(Paragraph("7.1 Web Local Yielding (AISC J10.2)", styles['SubsectionHeader']))
    story.append(Paragraph(f"For loads applied at member ends, per Eq. J10-3:", styles['BodyTextCustom']))
    story.append(Paragraph(f"Rn = Fy × tw × (2.5k + lb) = {Fy} × {sec.tw:.1f} × (2.5×{wly['k']:.1f} + 150) / 1000 = <b>{wly['Rn']:.2f} kN</b>", styles['Equation']))
    story.append(Paragraph(f"Allowable: Rn/Ω = {wly['Rn']:.2f} / 1.50 = <b>{wly['Rn_allow']:.2f} kN</b>", styles['Equation']))
    story.append(Paragraph(f"Applied reaction: R = {results.R_A_max:.2f} kN", styles['BodyTextCustom']))
    story.append(Paragraph(f"<b>Ratio = {results.R_A_max:.2f} / {wly['Rn_allow']:.2f} = {wly['ratio']:.3f}</b>", styles['Equation']))
    if wly['ratio'] <= 1.0:
        story.append(Paragraph(f"<b>WEB LOCAL YIELDING: PASS ✓</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"<b>WEB LOCAL YIELDING: FAIL - Bearing stiffeners required ✗</b>", styles['ResultFail']))
    
    story.append(Paragraph("7.2 Web Crippling (AISC J10.3)", styles['SubsectionHeader']))
    story.append(Paragraph(f"For loads applied at member ends, per Eq. J10-4:", styles['BodyTextCustom']))
    story.append(Paragraph(f"Rn = <b>{wcr['Rn']:.2f} kN</b>", styles['Equation']))
    story.append(Paragraph(f"Allowable: Rn/Ω = {wcr['Rn']:.2f} / 2.00 = <b>{wcr['Rn_allow']:.2f} kN</b>", styles['Equation']))
    story.append(Paragraph(f"<b>Ratio = {results.R_A_max:.2f} / {wcr['Rn_allow']:.2f} = {wcr['ratio']:.3f}</b>", styles['Equation']))
    if wcr['ratio'] <= 1.0:
        story.append(Paragraph(f"<b>WEB CRIPPLING: PASS ✓</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"<b>WEB CRIPPLING: FAIL - Bearing stiffeners required ✗</b>", styles['ResultFail']))
    
    # ===================== 8. WELD DESIGN (for built-up sections) =====================
    sect_num = 8
    if weld_check is not None:
        story.append(Paragraph(f"{sect_num}. WELD DESIGN FOR BUILT-UP SECTION", styles['SectionHeader']))
        story.append(Paragraph("<i>Reference: AISC 360-16 Chapter J2 - Welds; AWS D1.1 Structural Welding Code</i>", styles['Reference']))
        
        story.append(Paragraph(f"<b>Weld Type:</b> {WELD_TYPES[weld_check['weld_type']]['name']}", styles['BodyTextCustom']))
        story.append(Paragraph(f"<b>Electrode:</b> {weld_check['electrode']} (FEXX = {weld_check['FEXX']} MPa)", styles['BodyTextCustom']))
        story.append(Paragraph(f"<b>Weld Size:</b> {weld_check['weld_size']:.1f} mm", styles['BodyTextCustom']))
        
        story.append(Paragraph("8.1 Shear Flow Calculation", styles['SubsectionHeader']))
        story.append(Paragraph(
            "The horizontal shear flow at the flange-web junction transfers the shear force "
            "between the flange and web. Per mechanics of materials:",
            styles['BodyTextCustom']
        ))
        story.append(Paragraph(f"<b>q = V × Q / I</b>", styles['Equation']))
        story.append(Paragraph(
            f"Where:<br/>"
            f"• V = Design shear = {V_design:.2f} kN<br/>"
            f"• Q = First moment of flange area = A_f × ȳ = {weld_check['A_flange']:.0f} × {weld_check['y_flange']:.1f} = {weld_check['Q']:.0f} mm³<br/>"
            f"• I = Moment of inertia = {sec.Ix/1e6:.2f} × 10⁶ mm⁴",
            styles['BodyTextCustom']
        ))
        story.append(Paragraph(
            f"q = {V_design*1000:.0f} × {weld_check['Q']:.0f} / {sec.Ix:.0f} = <b>{weld_check['shear_flow']:.2f} N/mm</b>",
            styles['Equation']
        ))
        story.append(Paragraph(
            f"With 2 weld lines (both sides of web): q per weld = <b>{weld_check['q_per_weld']:.2f} N/mm</b>",
            styles['BodyTextCustom']
        ))
        
        story.append(Paragraph("8.2 Weld Size Requirements", styles['SubsectionHeader']))
        story.append(Paragraph(
            f"<b>Minimum size (Table J2.4):</b> w_min = {weld_check['w_min']:.1f} mm (based on t = {max(sec.tf_top if sec.tf_top > 0 else sec.tf, sec.tw):.1f} mm)",
            styles['BodyTextCustom']
        ))
        story.append(Paragraph(
            f"<b>Maximum size (J2.2b):</b> w_max = {weld_check['w_max']:.1f} mm",
            styles['BodyTextCustom']
        ))
        if 'w_required' in weld_check:
            story.append(Paragraph(
                f"<b>Required by calculation:</b> w_req = {weld_check['w_required']:.1f} mm",
                styles['BodyTextCustom']
            ))
        
        story.append(Paragraph("8.3 Weld Strength Check", styles['SubsectionHeader']))
        if weld_check['weld_type'] == 'fillet':
            story.append(Paragraph(
                f"Effective throat: a = 0.707 × w = 0.707 × {weld_check['weld_size']:.1f} = <b>{weld_check['throat']:.2f} mm</b>",
                styles['Equation']
            ))
            story.append(Paragraph(
                f"Nominal strength (Eq. J2-4): Rn = 0.60 × FEXX × a = 0.60 × {weld_check['FEXX']} × {weld_check['throat']:.2f} = <b>{weld_check['weld_strength']['Rn']:.2f} N/mm</b>",
                styles['Equation']
            ))
            story.append(Paragraph(
                f"Allowable strength (ASD): Rn/Ω = {weld_check['weld_strength']['Rn']:.2f} / 2.00 = <b>{weld_check['Rn_allow']:.2f} N/mm</b>",
                styles['Equation']
            ))
        
        story.append(Paragraph(
            f"<b>Demand/Capacity Ratio = {weld_check['q_per_weld']:.2f} / {weld_check['Rn_allow']:.2f} = {weld_check['ratio']:.3f}</b>",
            styles['Equation']
        ))
        
        if weld_check['ok']:
            story.append(Paragraph(f"<b>WELD DESIGN: PASS ✓</b>", styles['ResultPass']))
        else:
            story.append(Paragraph(f"<b>WELD DESIGN: FAIL ✗</b>", styles['ResultFail']))
        
        sect_num = 9
    
    # ===================== FATIGUE (if enabled) =====================
    if check_fatigue_enabled:
        story.append(Paragraph(f"{sect_num}. FATIGUE DESIGN", styles['SectionHeader']))
        story.append(Paragraph("<i>Reference: AISC 360-16, Appendix 3 - Design for Fatigue</i>", styles['Reference']))
        
        story.append(Paragraph(f"Crane Classification: CMAA Class {crane_class}", styles['BodyTextCustom']))
        story.append(Paragraph(f"Design Life: N = {N_cycles:,} cycles", styles['BodyTextCustom']))
        story.append(Paragraph(f"Fatigue Category: {fatigue_cat}", styles['BodyTextCustom']))
        story.append(Paragraph(f"Stress range: f_sr = M_range / Sx = {results.M_max:.2f}×10⁶ / {sec.Sx/1e3:.1f}×10³ = <b>{fatigue['f_sr']:.2f} MPa</b>", styles['Equation']))
        story.append(Paragraph(f"Allowable stress range (Eq. A-3-1): F_SR = (Cf/N)^(1/3) ≥ FTH = <b>{fatigue['F_sr']:.2f} MPa</b>", styles['Equation']))
        story.append(Paragraph(f"<b>Ratio = f_sr / F_SR = {fatigue['f_sr']:.2f} / {fatigue['F_sr']:.2f} = {fatigue['ratio']:.3f}</b>", styles['Equation']))
        if fatigue['ratio'] <= 1.0:
            story.append(Paragraph(f"<b>FATIGUE CHECK: PASS ✓</b>", styles['ResultPass']))
        else:
            story.append(Paragraph(f"<b>FATIGUE CHECK: FAIL ✗</b>", styles['ResultFail']))
        sect_num = 9
    
    # ===================== DEFLECTION =====================
    story.append(Paragraph(f"{sect_num}. DEFLECTION CHECK", styles['SectionHeader']))
    story.append(Paragraph("<i>Reference: AISC Design Guide 7, Table 4.1 - Recommended Deflection Limits</i>", styles['Reference']))
    
    defl_limit_ratio = int(beam_span * 1000 / delta_limit) if delta_limit > 0 else 600
    story.append(Paragraph(f"For CMAA Class {crane_class}, recommended vertical deflection limit:", styles['BodyTextCustom']))
    story.append(Paragraph(f"δ_allow = L/{defl_limit_ratio} = {beam_span*1000:.0f}/{defl_limit_ratio} = <b>{delta_limit:.2f} mm</b>", styles['Equation']))
    story.append(Paragraph(f"Calculated maximum deflection: δ = <b>{delta:.2f} mm</b>", styles['Equation']))
    story.append(Paragraph(f"<b>Ratio = δ / δ_allow = {delta:.2f} / {delta_limit:.2f} = {defl_ratio:.3f}</b>", styles['Equation']))
    if defl_ratio <= 1.0:
        story.append(Paragraph(f"<b>DEFLECTION CHECK: PASS ✓</b>", styles['ResultPass']))
    else:
        story.append(Paragraph(f"<b>DEFLECTION CHECK: FAIL ✗</b>", styles['ResultFail']))
    
    story.append(PageBreak())
    
    # ===================== DESIGN SUMMARY =====================
    sect_num = 10 if check_fatigue_enabled else 9
    story.append(Paragraph(f"{sect_num}. DESIGN SUMMARY", styles['SectionHeader']))
    
    summary_data = [['Design Check', 'Demand', 'Capacity', 'Ratio', 'Status'],
                    ['Flexure (Ch. F)', f'{M_design:.1f} kN-m', f'{flex["Mn_allow"]:.1f} kN-m', f'{flex_ratio:.3f}', '✓ PASS' if flex_ratio <= 1.0 else '✗ FAIL'],
                    ['Shear (Ch. G)', f'{V_design:.1f} kN', f'{shear["Vn_allow"]:.1f} kN', f'{shear_ratio:.3f}', '✓ PASS' if shear_ratio <= 1.0 else '✗ FAIL'],
                    ['Web Local Yielding (J10.2)', f'{results.R_A_max:.1f} kN', f'{wly["Rn_allow"]:.1f} kN', f'{wly["ratio"]:.3f}', '✓ PASS' if wly['ratio'] <= 1.0 else '✗ FAIL'],
                    ['Web Crippling (J10.3)', f'{results.R_A_max:.1f} kN', f'{wcr["Rn_allow"]:.1f} kN', f'{wcr["ratio"]:.3f}', '✓ PASS' if wcr['ratio'] <= 1.0 else '✗ FAIL'],
                    ['Deflection', f'{delta:.1f} mm', f'{delta_limit:.1f} mm', f'{defl_ratio:.3f}', '✓ PASS' if defl_ratio <= 1.0 else '✗ FAIL']]
    
    if check_fatigue_enabled:
        summary_data.append(['Fatigue (App. 3)', f'{fatigue["f_sr"]:.1f} MPa', f'{fatigue["F_sr"]:.1f} MPa', 
                           f'{fatigue["ratio"]:.3f}', '✓ PASS' if fatigue['ratio'] <= 1.0 else '✗ FAIL'])
    
    story.append(create_pdf_table(summary_data, col_widths=[45*mm, 28*mm, 28*mm, 20*mm, 22*mm]))
    story.append(Spacer(1, 10*mm))
    
    # Final conclusion
    if overall_pass:
        story.append(Paragraph(
            f"<b>CONCLUSION:</b><br/>The {sec.name} section is <b>ADEQUATE</b> for the design loads. "
            f"All strength and serviceability checks pass per AISC 360-16.",
            styles['ResultPass']
        ))
    else:
        story.append(Paragraph(
            f"<b>CONCLUSION:</b><br/>The {sec.name} section is <b>NOT ADEQUATE</b> for the design loads. "
            f"Revise the design by selecting a larger section or adding stiffeners.",
            styles['ResultFail']
        ))
    
    story.append(Spacer(1, 15*mm))
    
    # ===================== REFERENCES =====================
    story.append(Paragraph("REFERENCES", styles['SectionHeader']))
    refs = [
        "[1] AISC 360-16, Specification for Structural Steel Buildings, American Institute of Steel Construction, Chicago, IL, 2016.",
        "[2] AISC Design Guide 7: Industrial Buildings - Roofs to Anchor Rods, 2nd Edition, AISC, 2004.",
        "[3] CMAA Specification No. 70, Specifications for Top Running Bridge & Gantry Type Multiple Girder Electric Overhead Traveling Cranes, 2015.",
        "[4] CMAA Specification No. 74, Specifications for Top Running & Under Running Single Girder Electric Overhead Traveling Cranes, 2015.",
        "[5] Fisher, J.M., Industrial Buildings: Roofs to Anchor Rods, AISC Steel Design Guide Series, 2004.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['BodyTextCustom']))
        story.append(Spacer(1, 2*mm))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Page config
st.set_page_config(
    page_title="Crane Runway Beam Design Pro V6.0",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("🏗️ Crane Runway Beam Design Pro V6.0")
    st.markdown("**AISC 360-16 (ASD) | Design Guide 7 | CMAA 70/74**")
    
    # ========== SIDEBAR INPUTS ==========
    with st.sidebar:
        st.header("📋 Input Parameters")
        
        # Beam Data
        st.subheader("🔩 Beam Data")
        beam_span = st.number_input("Beam Span (m)", value=12.0, min_value=3.0, max_value=30.0, step=0.5)
        col1, col2 = st.columns(2)
        Lb = col1.number_input("Unbraced Lb (m)", value=beam_span, min_value=0.5, step=0.5)
        step_size = col2.number_input("Step Size (m)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
        
        # Material
        st.subheader("🔧 Material")
        steel_grade = st.selectbox("Steel Grade", list(STEEL_GRADES.keys()), index=2)
        Fy = STEEL_GRADES[steel_grade]['Fy']
        Fu = STEEL_GRADES[steel_grade]['Fu']
        st.caption(f"Fy = {Fy} MPa, Fu = {Fu} MPa")
        
        # Section Type
        st.subheader("📐 Section")
        sec_type = st.radio("Section Type", ["Hot Rolled", "Built-up Plate Girder"], horizontal=True)
        
        if sec_type == "Hot Rolled":
            sec_family = st.selectbox("Family", list(SECTION_DB.keys()), index=1)
            sec_name = st.selectbox("Section", list(SECTION_DB[sec_family].keys()), index=5)
            props = SECTION_DB[sec_family][sec_name]
            
            sec = Section(
                name=sec_name, sec_type='hot_rolled',
                d=props['d'], bf=props['bf'], tf=props['tf'],
                tw=props.get('tw', props['tf']/2), r=props.get('r', 0),
                A=props['A'], Ix=props['Ix'], Iy=props['Iy'],
                Sx=props['Sx'], Zx=props.get('Zx', props['Sx']*1.12),
                mass=props['mass']
            )
            sec.hw = sec.d - 2 * sec.tf
            sec.bf_top = sec.bf
            sec.tf_top = sec.tf
            sec.bf_bot = sec.bf
            sec.tf_bot = sec.tf
            sec.calc_props()
        else:
            st.markdown("**Plate Girder Dimensions (mm):**")
            col1, col2 = st.columns(2)
            bu_d = col1.number_input("Total Depth d", value=700.0, min_value=300.0, step=50.0)
            bu_tw = col2.number_input("Web tw", value=10.0, min_value=6.0, step=1.0)
            
            st.markdown("**Top Flange:**")
            col1, col2 = st.columns(2)
            bu_bf_top = col1.number_input("Width bf_top", value=250.0, min_value=100.0, step=10.0)
            bu_tf_top = col2.number_input("Thickness tf_top", value=16.0, min_value=8.0, step=1.0)
            
            st.markdown("**Bottom Flange:**")
            col1, col2 = st.columns(2)
            bu_bf_bot = col1.number_input("Width bf_bot", value=250.0, min_value=100.0, step=10.0)
            bu_tf_bot = col2.number_input("Thickness tf_bot", value=16.0, min_value=8.0, step=1.0)
            
            hw = bu_d - bu_tf_top - bu_tf_bot
            sec = Section(
                name=f"PG {bu_d:.0f}×{bu_tw:.0f}",
                sec_type='built_up', d=bu_d, tw=bu_tw, hw=hw,
                bf_top=bu_bf_top, tf_top=bu_tf_top,
                bf_bot=bu_bf_bot, tf_bot=bu_tf_bot
            )
            sec.calc_props()
        
        # Cap Channel
        st.subheader("🔗 Cap Channel")
        has_cap = st.checkbox("Add Cap Channel", value=False)
        if has_cap:
            cap_family = st.selectbox("Channel Type", list(CHANNEL_DB.keys()))
            cap_name = st.selectbox("Channel Size", list(CHANNEL_DB[cap_family].keys()), index=2)
            cap_props = CHANNEL_DB[cap_family][cap_name]
            
            sec.has_cap = True
            sec.cap_name = cap_name
            sec.cap_A = cap_props['A']
            sec.cap_Ix = cap_props['Ix']
            sec.cap_Iy = cap_props['Iy']
            sec.cap_d = cap_props['d']
            sec.cap_cy = cap_props.get('cy', cap_props['d']/2)
            sec.calc_props()
            st.caption(f"Cap: {cap_name}, A = {cap_props['A']} mm²")
        
        # Stiffeners
        st.subheader("🔩 Stiffeners")
        stiff = StiffenerData()
        
        stiff.has_transverse = st.checkbox("Transverse Stiffeners", value=False)
        if stiff.has_transverse:
            col1, col2, col3 = st.columns(3)
            stiff.trans_spacing = col1.number_input("Spacing", value=1500.0, min_value=100.0, key='ts_s')
            stiff.trans_t = col2.number_input("t", value=10.0, min_value=6.0, key='ts_t')
            stiff.trans_b = col3.number_input("b", value=80.0, min_value=30.0, key='ts_b')
        
        stiff.has_bearing = st.checkbox("Bearing Stiffeners", value=False)
        if stiff.has_bearing:
            col1, col2 = st.columns(2)
            stiff.bearing_t = col1.number_input("t", value=12.0, min_value=8.0, key='bs_t')
            stiff.bearing_b = col2.number_input("b", value=100.0, min_value=50.0, key='bs_b')
        
        stiff.has_longitudinal = st.checkbox("Longitudinal Stiffeners", value=False)
        if stiff.has_longitudinal:
            col1, col2 = st.columns(2)
            stiff.long_t = col1.number_input("t", value=10.0, min_value=6.0, key='ls_t')
            stiff.long_b = col2.number_input("b", value=80.0, min_value=30.0, key='ls_b')
            stiff.long_position = st.slider("Position", 0.15, 0.35, 0.2)
        
        # Weld Design (for Built-up Sections)
        weld_data = None
        if sec_type == "Built-up Plate Girder":
            st.subheader("🔥 Weld Design")
            check_weld = st.checkbox("Design Flange-Web Welds", value=True)
            
            if check_weld:
                weld_type = st.selectbox(
                    "Weld Type",
                    options=list(WELD_TYPES.keys()),
                    format_func=lambda x: WELD_TYPES[x]['name'],
                    index=0
                )
                st.caption(WELD_TYPES[weld_type]['desc'])
                
                electrode = st.selectbox(
                    "Electrode",
                    options=list(WELD_ELECTRODES.keys()),
                    format_func=lambda x: f"{x} (FEXX = {WELD_ELECTRODES[x]['FEXX']} MPa)",
                    index=1  # E70 as default
                )
                
                col1, col2 = st.columns(2)
                if weld_type == 'fillet':
                    weld_size = col1.number_input(
                        "Leg Size (mm)", value=6.0, min_value=3.0, max_value=25.0, step=1.0,
                        help="Fillet weld leg dimension"
                    )
                else:
                    weld_size = col1.number_input(
                        "Throat (mm)", value=10.0, min_value=5.0, max_value=50.0, step=1.0,
                        help="Effective throat dimension"
                    )
                
                num_welds = col2.selectbox("Weld Lines", [2, 4], index=0, 
                                           help="2 = both sides of web, 4 = both flanges both sides")
                
                is_continuous = st.checkbox("Continuous Weld", value=True)
                
                intermittent_length = 0.0
                intermittent_spacing = 0.0
                if not is_continuous:
                    col1, col2 = st.columns(2)
                    intermittent_length = col1.number_input("Segment Length (mm)", value=50.0, min_value=25.0)
                    intermittent_spacing = col2.number_input("Spacing c/c (mm)", value=150.0, min_value=50.0)
                
                weld_data = WeldDesignData(
                    weld_type=weld_type,
                    electrode=electrode,
                    weld_size=weld_size,
                    num_welds=num_welds,
                    is_continuous=is_continuous,
                    intermittent_length=intermittent_length,
                    intermittent_spacing=intermittent_spacing
                )
        
        # Crane Data
        st.subheader("🏗️ Crane Data")
        num_cranes = st.radio("Number of Cranes", [1, 2, 3], horizontal=True)
        
        cranes = []
        for i in range(1, num_cranes + 1):
            with st.expander(f"Crane {i}", expanded=(i == 1)):
                use_direct = st.checkbox("Direct Input", key=f"direct_{i}")
                
                if use_direct:
                    col1, col2 = st.columns(2)
                    d_max = col1.number_input("Max Wheel (kN)", value=62.0, key=f"dmax_{i}")
                    d_min = col2.number_input("Min Wheel (kN)", value=13.0, key=f"dmin_{i}")
                    d_lat = col1.number_input("Lateral/Wheel (kN)", value=5.0, key=f"dlat_{i}")
                    cap = col2.number_input("Capacity (T)", value=10.0, key=f"cap_{i}")
                    bw, tw_c, bs, mha = 5.0, 0.72, 20.0, 1.0
                else:
                    col1, col2 = st.columns(2)
                    cap = col1.number_input("Capacity (T)", value=10.0, key=f"cap_{i}")
                    bw = col2.number_input("Bridge Wt (T)", value=5.0, key=f"bw_{i}")
                    tw_c = col1.number_input("Trolley Wt (T)", value=0.72, key=f"tw_{i}")
                    bs = col2.number_input("Bridge Span (m)", value=20.0, key=f"bs_{i}")
                    mha = st.number_input("Min Hook Approach (m)", value=1.0, key=f"mha_{i}")
                    d_max, d_min, d_lat = 0, 0, 0
                
                col1, col2 = st.columns(2)
                nw = col1.selectbox("Wheels per Rail", [2, 4], index=0, key=f"nw_{i}")
                
                # Wheel spacing configuration
                ws_12, ws_23, ws_34 = 0.0, 0.0, 0.0
                
                if nw == 2:
                    wb = col2.number_input("Wheel Base (m)", value=3.15, min_value=0.5, key=f"wb_{i}",
                                          help="Distance between the two wheels")
                else:  # 4 wheels
                    st.markdown("**4-Wheel Configuration:**")
                    st.caption("W1 --[d12]-- W2 --[d23]-- W3 --[d34]-- W4")
                    col1, col2, col3 = st.columns(3)
                    ws_12 = col1.number_input("d12 (m)", value=1.0, min_value=0.3, key=f"ws12_{i}",
                                             help="Distance: Wheel 1 to Wheel 2")
                    ws_23 = col2.number_input("d23 (m)", value=2.5, min_value=0.3, key=f"ws23_{i}",
                                             help="Distance: Wheel 2 to Wheel 3 (center gap)")
                    ws_34 = col3.number_input("d34 (m)", value=1.0, min_value=0.3, key=f"ws34_{i}",
                                             help="Distance: Wheel 3 to Wheel 4")
                    wb = ws_12 + ws_23 + ws_34  # Total wheel base
                    st.caption(f"Total wheel base: {wb:.3f} m")
                
                col1, col2 = st.columns(2)
                buf_l = col1.number_input("Buffer L (m)", value=0.266, key=f"bufl_{i}")
                buf_r = col2.number_input("Buffer R (m)", value=0.266, key=f"bufr_{i}")
                
                col1, col2, col3 = st.columns(3)
                iv = col1.number_input("V%", value=25, key=f"iv_{i}")
                ih = col2.number_input("H%", value=20, key=f"ih_{i}")
                il = col3.number_input("L%", value=10, key=f"il_{i}")
                
                crane = CraneData(
                    crane_id=i, capacity_tonnes=cap, bridge_weight=bw, trolley_weight=tw_c,
                    bridge_span=bs, min_hook_approach=mha, wheel_base=wb,
                    buffer_left=buf_l, buffer_right=buf_r, num_wheels=nw,
                    impact_v=iv/100, impact_h=ih/100, impact_l=il/100,
                    use_direct_input=use_direct, direct_max_wheel_load=d_max,
                    direct_min_wheel_load=d_min, direct_lateral_load=d_lat,
                    wheel_spacing_12=ws_12, wheel_spacing_23=ws_23, wheel_spacing_34=ws_34
                )
                cranes.append(crane)
                
                # Show wheel positions
                wheel_pos = crane.get_wheel_positions_relative()
                pos_str = ", ".join([f"W{j+1}:{p:.3f}m" for j, p in enumerate(wheel_pos)])
                st.caption(f"Wheel positions: {pos_str}")
                st.caption(f"Max wheel load: {crane.get_wheel_load_with_impact():.1f} kN (with impact)")
        
        # Design Parameters
        st.subheader("⚙️ Design Settings")
        crane_class = st.selectbox("Crane Class", list(CRANE_CLASSES.keys()), index=2,
                                  format_func=lambda x: f"{x} - {CRANE_CLASSES[x]['name']}")
        check_fatigue_enabled = st.checkbox("✓ Include Fatigue Check", value=True)
        if check_fatigue_enabled:
            fatigue_cat = st.selectbox("Fatigue Category", list(FATIGUE_CATEGORIES.keys()), index=4)
        else:
            fatigue_cat = 'E'
        
        # Option to include fatigue in PDF report
        include_fatigue_in_pdf = st.checkbox("✓ Include Fatigue in PDF Report", value=True, 
                                             disabled=not check_fatigue_enabled)
        if not check_fatigue_enabled:
            include_fatigue_in_pdf = False
        
        lb_bearing = st.number_input("Bearing Length (mm)", value=150.0, min_value=50.0)
        
        st.markdown("---")
        run_design = st.button("🚀 RUN DESIGN", type="primary", use_container_width=True)
    
    # ========== CHECK IF DESIGN SHOULD RUN ==========
    if 'design_completed' not in st.session_state:
        st.session_state.design_completed = False
    
    if run_design:
        st.session_state.design_completed = True
    
    if not st.session_state.design_completed:
        st.info("👈 Configure parameters in the sidebar and click **RUN DESIGN** to start analysis.")
        # Preview section
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Section Preview")
            fig_preview = plot_section(sec, stiff)
            st.plotly_chart(fig_preview, use_container_width=True)
        with col2:
            st.markdown("### Section Properties")
            st.write(f"**{sec.name}**" + (f" + {sec.cap_name}" if sec.has_cap else ""))
            st.write(f"d = {sec.d:.0f} mm | bf = {max(sec.bf_top, sec.bf_bot) if sec.bf_top > 0 else sec.bf:.0f} mm")
            st.write(f"tf = {sec.tf_top if sec.tf_top > 0 else sec.tf:.1f} mm | tw = {sec.tw:.1f} mm")
            st.write(f"Ix = {sec.Ix/1e6:.2f} ×10⁶ mm⁴ | Sx = {sec.Sx/1e3:.1f} ×10³ mm³")
            st.write(f"Mass = {sec.mass:.1f} kg/m")
        return
    
    # ========== RUN ANALYSIS ==========
    results = run_moving_load_analysis(beam_span, cranes, step_size, num_cranes)
    
    # Self-weight
    w_self = sec.mass * GRAVITY / 1000
    M_self = w_self * beam_span**2 / 8
    V_self = w_self * beam_span / 2
    R_self = V_self
    
    # Design values
    M_design = results.M_max + M_self
    V_design = results.V_max + V_self
    
    # Design checks
    cmp = check_compactness(sec, Fy)
    flex = calc_flexural_strength(sec, Fy, Lb * 1000, cmp)
    shear = calc_shear_strength(sec, Fy, stiff.has_transverse, stiff.trans_spacing)
    wly = check_web_local_yielding(sec, Fy, results.R_A_max, lb_bearing, True)
    wcr = check_web_crippling(sec, Fy, results.R_A_max, lb_bearing, True)
    
    N_cycles = CRANE_CLASSES[crane_class]['max_cycles']
    if check_fatigue_enabled:
        fatigue = check_fatigue(sec, results.M_max, N_cycles, fatigue_cat)
    else:
        fatigue = {'ratio': 0, 'f_sr': 0, 'F_sr': 999, 'status': 'N/A', 'FTH': 0, 'category': 'N/A'}
    
    # Stiffener checks
    trans_check = check_transverse_stiffener(sec, Fy, stiff)
    bearing_check = check_bearing_stiffener(sec, Fy, results.R_A_max, stiff, True)
    long_check = check_longitudinal_stiffener(sec, Fy, stiff)
    
    # Weld check (for built-up sections)
    weld_check = None
    if weld_data is not None and sec.sec_type == 'built_up':
        weld_check = check_weld_for_built_up_section(sec, V_design, M_design, weld_data, Fy)
    
    # Deflection
    if results.M_max_wheel_positions:
        delta = calc_deflection(
            results.all_results[0].wheel_loads,
            results.M_max_wheel_positions, beam_span, E_STEEL, sec.Ix
        )
    else:
        delta = 0
    delta_limit = beam_span * 1000 / CRANE_CLASSES[crane_class]['defl_limit']
    
    # Ratios
    flex_ratio = M_design / flex['Mn_allow'] if flex['Mn_allow'] > 0 else 999
    shear_ratio = V_design / shear['Vn_allow'] if shear['Vn_allow'] > 0 else 999
    defl_ratio = delta / delta_limit if delta_limit > 0 else 0
    
    # ========== DISPLAY TABS ==========
    tab_names = ["📊 Summary", "📈 Moving Loads", "🔍 Design Checks", "🔄 Fatigue", "🔩 Stiffeners", "📐 Bracket Loads", "📄 Report"]
    if weld_check is not None:
        tab_names.insert(5, "🔥 Weld Design")
    tabs = st.tabs(tab_names)
    
    # TAB 1: SUMMARY
    with tabs[0]:
        st.subheader("Design Summary")
        
        checks_list = [flex_ratio, shear_ratio, wly['ratio'], wcr['ratio'], defl_ratio]
        if check_fatigue_enabled:
            checks_list.append(fatigue['ratio'])
        if weld_check is not None:
            checks_list.append(weld_check['ratio'])
        
        all_ok = all(r <= 1.0 for r in checks_list)
        
        if all_ok:
            st.success("✅ **ALL CHECKS PASS** - Section is adequate")
        else:
            st.error("❌ **SOME CHECKS FAIL** - Revise section")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("M_design", f"{M_design:.0f} kN-m", f"Ratio: {flex_ratio:.3f}")
        col2.metric("V_design", f"{V_design:.0f} kN", f"Ratio: {shear_ratio:.3f}")
        col3.metric("δ_max", f"{delta:.1f} mm", f"Limit: {delta_limit:.1f}")
        if check_fatigue_enabled:
            col4.metric("Fatigue", f"{fatigue['f_sr']:.0f} MPa", f"Ratio: {fatigue['ratio']:.3f}")
        else:
            col4.metric("Fatigue", "Not Checked", "Disabled")
        
        # Add weld check metric if applicable
        if weld_check is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Weld Shear", f"{weld_check['q_per_weld']:.1f} N/mm", f"Ratio: {weld_check['ratio']:.3f}")
        
        st.markdown("---")
        
        # Build data table dynamically
        checks = ['Flexure', 'Shear', 'Web Yielding', 'Web Crippling']
        demands = [f"{M_design:.0f}", f"{V_design:.0f}", f"{results.R_A_max:.0f}", f"{results.R_A_max:.0f}"]
        capacities = [f"{flex['Mn_allow']:.0f}", f"{shear['Vn_allow']:.0f}", f"{wly['Rn_allow']:.0f}", f"{wcr['Rn_allow']:.0f}"]
        units = ['kN-m', 'kN', 'kN', 'kN']
        ratios = [f"{flex_ratio:.3f}", f"{shear_ratio:.3f}", f"{wly['ratio']:.3f}", f"{wcr['ratio']:.3f}"]
        statuses = ['✅' if r <= 1.0 else '❌' for r in [flex_ratio, shear_ratio, wly['ratio'], wcr['ratio']]]
        
        # Add fatigue
        checks.append('Fatigue')
        demands.append(f"{fatigue['f_sr']:.1f}")
        capacities.append(f"{fatigue['F_sr']:.1f}")
        units.append('MPa')
        ratios.append(f"{fatigue['ratio']:.3f}")
        statuses.append('✅' if fatigue['ratio'] <= 1.0 else '❌' if check_fatigue_enabled else '⚪')
        
        # Add deflection
        checks.append('Deflection')
        demands.append(f"{delta:.1f}")
        capacities.append(f"{delta_limit:.1f}")
        units.append('mm')
        ratios.append(f"{defl_ratio:.3f}")
        statuses.append('✅' if defl_ratio <= 1.0 else '❌')
        
        # Add weld if applicable
        if weld_check is not None:
            checks.append('Weld (Shear Flow)')
            demands.append(f"{weld_check['q_per_weld']:.1f}")
            capacities.append(f"{weld_check['Rn_allow']:.1f}")
            units.append('N/mm')
            ratios.append(f"{weld_check['ratio']:.3f}")
            statuses.append('✅' if weld_check['ok'] else '❌')
        
        data = {
            'Check': checks,
            'Demand': demands,
            'Capacity': capacities,
            'Unit': units,
            'Ratio': ratios,
            'Status': statuses
        }
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Section Properties:**")
            st.write(f"• Name: **{sec.name}**" + (f" + {sec.cap_name}" if sec.has_cap else ""))
            st.write(f"• d = {sec.d:.0f} mm, tw = {sec.tw:.1f} mm")
            st.write(f"• Ix = {sec.Ix/1e6:.1f} ×10⁶ mm⁴")
            st.write(f"• Sx = {sec.Sx/1e3:.0f} ×10³ mm³")
            st.write(f"• Mass = {sec.mass:.1f} kg/m")
        with col2:
            fig_sec = plot_section(sec, stiff)
            st.plotly_chart(fig_sec, use_container_width=True)
    
    # TAB 2: MOVING LOADS ANALYSIS
    with tabs[1]:
        st.subheader("Moving Load Analysis - Complete Results")
        
        # Analysis parameters
        st.markdown(f"""
        **Analysis Parameters:**
        - Beam Span: L = {beam_span} m
        - Step Size: {step_size} m  
        - Total Analysis Steps: {len(results.all_results)}
        - Number of Cranes: {num_cranes}
        - Total Wheels: {sum(c.num_wheels for c in cranes)}
        """)
        
        # Influence diagrams
        st.markdown("### Influence Diagrams")
        fig_inf = plot_influence_diagrams(results, beam_span)
        st.plotly_chart(fig_inf, use_container_width=True)
        
        # Critical values with details
        st.markdown("### Critical Values")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔵 Maximum Moment")
            st.metric("M_max", f"{results.M_max:.2f} kN-m")
            st.write(f"**Location:** x = {results.M_max_location:.3f} m")
            st.write(f"**First wheel at:** {results.M_max_position:.3f} m")
            st.write(f"**Reactions:** R_A = {results.M_max_R_A:.2f} kN, R_B = {results.M_max_R_B:.2f} kN")
            st.write(f"**Wheel positions:** {', '.join([f'{p:.2f}' for p in results.M_max_wheel_positions])} m")
        
        with col2:
            st.markdown("#### 🟢 Maximum Left Reaction")
            st.metric("R_A,max", f"{results.R_A_max:.2f} kN")
            st.write(f"**First wheel at:** {results.R_A_max_position:.3f} m")
            st.write(f"**Corresponding R_B:** {results.R_A_max_R_B:.2f} kN")
            st.write(f"**Corresponding M:** {results.R_A_max_moment:.2f} kN-m")
            st.write(f"**Wheel positions:** {', '.join([f'{p:.2f}' for p in results.R_A_max_wheel_positions])} m")
        
        with col3:
            st.markdown("#### 🔴 Maximum Right Reaction")
            st.metric("R_B,max", f"{results.R_B_max:.2f} kN")
            st.write(f"**First wheel at:** {results.R_B_max_position:.3f} m")
            st.write(f"**Corresponding R_A:** {results.R_B_max_R_A:.2f} kN")
            st.write(f"**Corresponding M:** {results.R_B_max_moment:.2f} kN-m")
            st.write(f"**Wheel positions:** {', '.join([f'{p:.2f}' for p in results.R_B_max_wheel_positions])} m")
        
        # Wheel Position Sketch for Maximum Moment
        st.markdown("### 🎯 Wheel Position at Maximum Moment")
        fig_wheels = go.Figure()
        
        # Draw beam
        fig_wheels.add_trace(go.Scatter(
            x=[0, beam_span], y=[0, 0],
            mode='lines', line=dict(color='#2C3E50', width=8),
            name='Beam', showlegend=True
        ))
        
        # Draw supports
        fig_wheels.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(symbol='triangle-up', size=20, color='#27AE60'),
            name='Support A'
        ))
        fig_wheels.add_trace(go.Scatter(
            x=[beam_span], y=[0], mode='markers',
            marker=dict(symbol='triangle-up', size=20, color='#E74C3C'),
            name='Support B'
        ))
        
        # Draw wheels at max moment position
        wheel_y = 0.3
        for i, pos in enumerate(results.M_max_wheel_positions):
            fig_wheels.add_trace(go.Scatter(
                x=[pos], y=[wheel_y], mode='markers+text',
                marker=dict(symbol='circle', size=25, color='#3498DB', line=dict(color='#2C3E50', width=2)),
                text=[f'P{i+1}'], textposition='top center',
                name=f'Wheel {i+1}' if i == 0 else None, showlegend=(i == 0)
            ))
            # Load arrow
            fig_wheels.add_annotation(
                x=pos, y=wheel_y + 0.15, ax=pos, ay=wheel_y + 0.5,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#E74C3C'
            )
        
        # Mark moment location
        fig_wheels.add_trace(go.Scatter(
            x=[results.M_max_location], y=[-0.15], mode='markers+text',
            marker=dict(symbol='star', size=20, color='#F39C12'),
            text=['M_max'], textposition='bottom center',
            name='Max Moment Location'
        ))
        
        # Reaction arrows
        fig_wheels.add_annotation(x=0, y=-0.1, ax=0, ay=-0.4, showarrow=True, arrowhead=2, arrowcolor='#27AE60')
        fig_wheels.add_annotation(x=0, y=-0.5, text=f'R_A={results.M_max_R_A:.1f} kN', showarrow=False, font=dict(size=10))
        fig_wheels.add_annotation(x=beam_span, y=-0.1, ax=beam_span, ay=-0.4, showarrow=True, arrowhead=2, arrowcolor='#E74C3C')
        fig_wheels.add_annotation(x=beam_span, y=-0.5, text=f'R_B={results.M_max_R_B:.1f} kN', showarrow=False, font=dict(size=10))
        
        # Dimensions
        fig_wheels.add_annotation(x=beam_span/2, y=-0.7, text=f'L = {beam_span} m', showarrow=False, font=dict(size=12, color='#7F8C8D'))
        
        fig_wheels.update_layout(
            title=f'Wheel Configuration at Maximum Moment (M_max = {results.M_max:.2f} kN-m)',
            xaxis=dict(title='Position along beam (m)', range=[-0.5, beam_span+0.5], showgrid=True),
            yaxis=dict(range=[-0.8, 0.8], showticklabels=False, showgrid=False),
            height=350, showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_wheels, use_container_width=True)
        
        # Complete Analysis Steps Table
        st.markdown("### 📋 Complete Analysis Steps - All Reactions")
        st.caption("This table shows R_A, R_B, M_max for every analysis step")
        
        steps_data = []
        for i, r in enumerate(results.all_results):
            steps_data.append({
                'Step': i + 1,
                'First Wheel (m)': f"{r.step_position:.3f}",
                'R_A (kN)': f"{r.R_A:.2f}",
                'R_B (kN)': f"{r.R_B:.2f}",
                'V_max (kN)': f"{max(r.R_A, r.R_B):.2f}",
                'M_max (kN-m)': f"{r.M_max:.2f}",
                'M_location (m)': f"{r.M_max_location:.3f}",
                'Wheel Positions (m)': ', '.join([f'{p:.2f}' for p in r.wheel_positions])
            })
        
        df_steps = pd.DataFrame(steps_data)
        st.dataframe(df_steps, hide_index=True, use_container_width=True, height=400)
        
        # Download button for steps data
        csv_data = df_steps.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis Steps (CSV)",
            data=csv_data,
            file_name=f"moving_load_analysis_steps.csv",
            mime="text/csv"
        )
    
    # TAB 3: DESIGN CHECKS - COMPREHENSIVE
    with tabs[2]:
        st.subheader("Design Checks - AISC 360-16 (ASD)")
        
        # Section Classification
        st.markdown("### 1️⃣ Section Classification (Table B4.1b)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### Flange Classification: **{cmp['flange_class']}**")
            flange_color = '#27AE60' if cmp['flange_class'] == 'Compact' else ('#F39C12' if cmp['flange_class'] == 'Noncompact' else '#E74C3C')
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | λf = bf/(2tf) | **{cmp['lambda_f']:.2f}** |
            | λpf = 0.38√(E/Fy) | {cmp['lambda_pf']:.2f} |
            | λrf = 1.0√(E/Fy) | {cmp['lambda_rf']:.2f} |
            """)
            if cmp['lambda_f'] <= cmp['lambda_pf']:
                st.success(f"λf = {cmp['lambda_f']:.2f} ≤ λpf = {cmp['lambda_pf']:.2f} → **Compact**")
            elif cmp['lambda_f'] <= cmp['lambda_rf']:
                st.warning(f"λpf < λf ≤ λrf → **Noncompact**")
            else:
                st.error(f"λf > λrf → **Slender**")
        
        with col2:
            st.markdown(f"#### Web Classification: **{cmp['web_class']}**")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | λw = h/tw | **{cmp['lambda_w']:.2f}** |
            | λpw = 3.76√(E/Fy) | {cmp['lambda_pw']:.2f} |
            | λrw = 5.70√(E/Fy) | {cmp['lambda_rw']:.2f} |
            """)
            if cmp['lambda_w'] <= cmp['lambda_pw']:
                st.success(f"λw = {cmp['lambda_w']:.2f} ≤ λpw = {cmp['lambda_pw']:.2f} → **Compact**")
            elif cmp['lambda_w'] <= cmp['lambda_rw']:
                st.warning(f"λpw < λw ≤ λrw → **Noncompact**")
            else:
                st.error(f"λw > λrw → **Slender**")
        
        st.markdown("---")
        
        # Flexural Strength
        st.markdown("### 2️⃣ Flexural Strength (Chapter F)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Calculation Steps:")
            st.markdown(f"""
            **Step 1: Plastic Moment**
            - Mp = Fy × Zx = {Fy} × {sec.Zx/1e3:.1f}×10³ = **{flex['Mp']:.2f} kN-m**
            - My = Fy × Sx = {Fy} × {sec.Sx/1e3:.1f}×10³ = **{flex['My']:.2f} kN-m**
            
            **Step 2: Lateral Bracing Limits**
            - ry = {sec.ry:.2f} mm
            - Lp = 1.76 × ry × √(E/Fy) = **{flex['Lp']/1000:.3f} m**
            - Lr = **{flex['Lr']/1000:.3f} m**
            - Lb = **{Lb:.2f} m** (unbraced length)
            
            **Step 3: Determine Zone**
            """)
            if Lb*1000 <= flex['Lp']:
                st.success(f"Lb = {Lb:.2f} m ≤ Lp = {flex['Lp']/1000:.3f} m → **Zone 1: Yielding**")
            elif Lb*1000 <= flex['Lr']:
                st.warning(f"Lp < Lb ≤ Lr → **Zone 2: Inelastic LTB**")
            else:
                st.error(f"Lb = {Lb:.2f} m > Lr = {flex['Lr']/1000:.3f} m → **Zone 3: Elastic LTB**")
        
        with col2:
            st.markdown("#### Results:")
            st.markdown(f"""
            **Governing Limit State:** {flex['limit_state']}
            
            | Parameter | Value |
            |-----------|-------|
            | Cb | {flex['Cb']:.2f} |
            | Mn | {flex['Mn']:.2f} kN-m |
            | Ωb | {OMEGA_FLEX} |
            | **Mn/Ωb** | **{flex['Mn_allow']:.2f} kN-m** |
            """)
            
            st.markdown("#### Design Check:")
            st.markdown(f"""
            - M_design = M_crane + M_self = {results.M_max:.2f} + {M_self:.2f} = **{M_design:.2f} kN-m**
            - Mn/Ω = **{flex['Mn_allow']:.2f} kN-m**
            - **Ratio = {M_design:.2f} / {flex['Mn_allow']:.2f} = {flex_ratio:.3f}**
            """)
            if flex_ratio <= 1.0:
                st.success(f"✅ Ratio = {flex_ratio:.3f} ≤ 1.0 → **OK**")
            else:
                st.error(f"❌ Ratio = {flex_ratio:.3f} > 1.0 → **NG - Increase section**")
        
        st.markdown("---")
        
        # Shear Strength
        st.markdown("### 3️⃣ Shear Strength (Chapter G)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Calculation Steps:")
            hw = sec.hw if sec.hw > 0 else sec.d - 2 * sec.tf
            st.markdown(f"""
            **Step 1: Web Area**
            - hw = {hw:.1f} mm
            - tw = {sec.tw:.1f} mm
            - Aw = hw × tw = **{shear['Aw']:.0f} mm²**
            
            **Step 2: Slenderness Ratio**
            - h/tw = {shear['h_tw']:.2f}
            
            **Step 3: Shear Buckling Coefficient**
            - kv = {shear['kv']:.2f}
            - 1.10√(kv×E/Fy) = {1.10 * math.sqrt(shear['kv'] * E_STEEL / Fy):.2f}
            - 1.37√(kv×E/Fy) = {1.37 * math.sqrt(shear['kv'] * E_STEEL / Fy):.2f}
            
            **Step 4: Shear Coefficient**
            - Cv1 = **{shear['Cv1']:.4f}**
            """)
        
        with col2:
            st.markdown("#### Results:")
            st.markdown(f"""
            **Governing Limit State:** {shear['limit_state']}
            
            | Parameter | Value |
            |-----------|-------|
            | Aw | {shear['Aw']:.0f} mm² |
            | kv | {shear['kv']:.2f} |
            | Cv1 | {shear['Cv1']:.4f} |
            | Vn = 0.6×Fy×Aw×Cv1 | {shear['Vn']:.2f} kN |
            | Ωv | {OMEGA_SHEAR} |
            | **Vn/Ωv** | **{shear['Vn_allow']:.2f} kN** |
            """)
            
            st.markdown("#### Design Check:")
            st.markdown(f"""
            - V_design = V_crane + V_self = {results.V_max:.2f} + {V_self:.2f} = **{V_design:.2f} kN**
            - Vn/Ω = **{shear['Vn_allow']:.2f} kN**
            - **Ratio = {V_design:.2f} / {shear['Vn_allow']:.2f} = {shear_ratio:.3f}**
            """)
            if shear_ratio <= 1.0:
                st.success(f"✅ Ratio = {shear_ratio:.3f} ≤ 1.0 → **OK**")
            else:
                st.error(f"❌ Ratio = {shear_ratio:.3f} > 1.0 → **NG**")
        
        st.markdown("---")
        
        # Web Local Effects
        st.markdown("### 4️⃣ Web Local Effects (Section J10)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Web Local Yielding (J10.2)")
            st.markdown(f"""
            **At Support (end reaction):**
            
            | Parameter | Value |
            |-----------|-------|
            | k = tf + r | {wly['k']:.1f} mm |
            | lb (bearing length) | {lb_bearing:.0f} mm |
            | Rn = Fy×tw×(2.5k + lb) | {wly['Rn']:.2f} kN |
            | Ω | {OMEGA_WLY} |
            | **Rn/Ω** | **{wly['Rn_allow']:.2f} kN** |
            
            **Check:**
            - R_applied = {results.R_A_max:.2f} kN
            - **Ratio = {wly['ratio']:.3f}**
            """)
            if wly['ratio'] <= 1.0:
                st.success(f"✅ {wly['status']}")
            else:
                st.error(f"❌ {wly['status']} - Bearing stiffeners required")
        
        with col2:
            st.markdown("#### Web Crippling (J10.3)")
            tf_use = sec.tf if sec.tf > 0 else (sec.tf_top if sec.tf_top > 0 else sec.tf_bot)
            st.markdown(f"""
            **At Support (concentrated load):**
            
            | Parameter | Value |
            |-----------|-------|
            | lb/d | {lb_bearing/sec.d:.4f} |
            | tw/tf | {sec.tw/tf_use:.3f} |
            | Rn | {wcr['Rn']:.2f} kN |
            | Ω | {OMEGA_WCR} |
            | **Rn/Ω** | **{wcr['Rn_allow']:.2f} kN** |
            
            **Check:**
            - R_applied = {results.R_A_max:.2f} kN
            - **Ratio = {wcr['ratio']:.3f}**
            """)
            if wcr['ratio'] <= 1.0:
                st.success(f"✅ {wcr['status']}")
            else:
                st.error(f"❌ {wcr['status']} - Bearing stiffeners required")
        
        st.markdown("---")
        
        # Deflection Check
        st.markdown("### 5️⃣ Deflection Check")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Crane Class:** {crane_class} - {CRANE_CLASSES[crane_class]['name']}
            
            **Deflection Limit per CMAA/Design Guide 7:**
            - L/{CRANE_CLASSES[crane_class]['defl_limit']} = {beam_span*1000:.0f}/{CRANE_CLASSES[crane_class]['defl_limit']} = **{delta_limit:.2f} mm**
            """)
        with col2:
            st.markdown(f"""
            **Calculated Deflection:**
            - δ_max = **{delta:.2f} mm**
            
            **Check:**
            - Ratio = {delta:.2f} / {delta_limit:.2f} = **{defl_ratio:.3f}**
            """)
            if defl_ratio <= 1.0:
                st.success(f"✅ δ = {delta:.2f} mm ≤ {delta_limit:.2f} mm → **OK**")
            else:
                st.error(f"❌ δ = {delta:.2f} mm > {delta_limit:.2f} mm → **NG - Increase I**")
    
    # TAB 4: FATIGUE CHECK
    with tabs[3]:
        st.subheader("🔄 Fatigue Design Check - AISC 360-16 Appendix 3")
        
        if not check_fatigue_enabled:
            st.warning("⚠️ Fatigue check is disabled. Enable it in the sidebar to perform fatigue analysis.")
            st.info("""
            **When to check fatigue:**
            - Crane runway beams experience repeated load cycles
            - CMAA Class C and above typically require fatigue checks
            - Critical for welded connections (Category D, E)
            """)
        else:
            # Fatigue Theory
            st.markdown("### 📚 Fatigue Design Theory")
            st.markdown("""
            Per AISC 360-16 Appendix 3, fatigue design ensures that the stress range from 
            repeated loading does not exceed the allowable stress range for the connection detail.
            
            **Key Equation:** f_sr ≤ F_SR
            
            Where:
            - f_sr = computed stress range (from moment range)
            - F_SR = allowable stress range (from fatigue category and cycles)
            """)
            
            st.markdown("---")
            
            # Fatigue Parameters
            st.markdown("### 1️⃣ Design Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Crane Service:**
                - Crane Class: **{crane_class}** - {CRANE_CLASSES[crane_class]['name']}
                - Design Life Cycles: **N = {N_cycles:,}**
                
                **Fatigue Category:** **{fatigue_cat}**
                - Description: {FATIGUE_CATEGORIES[fatigue_cat]['desc']}
                - Cf = {FATIGUE_CATEGORIES[fatigue_cat]['Cf']:.2e} MPa³
                - FTH (threshold) = {FATIGUE_CATEGORIES[fatigue_cat]['FTH']} MPa
                """)
            
            with col2:
                # Fatigue category table
                st.markdown("**Common Fatigue Categories for Runway Beams:**")
                cat_data = {
                    'Category': ['B', 'C', 'D', 'E', "E'"],
                    'Detail': ['Rolled sections', 'Stiffener welds', 'Longer attachments', 'Fillet welds', 'Severe'],
                    'Cf (×10⁸)': [120, 44, 22, 11, 3.9],
                    'FTH (MPa)': [110, 69, 48, 31, 18]
                }
                st.dataframe(pd.DataFrame(cat_data), hide_index=True)
            
            st.markdown("---")
            
            # Stress Range Calculation
            st.markdown("### 2️⃣ Stress Range Calculation")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Moment Range:**
                - M_max = {results.M_max:.2f} kN-m (crane at critical position)
                - M_min ≈ 0 kN-m (crane off beam)
                - **M_range = {results.M_max:.2f} kN-m**
                
                **Section Modulus:**
                - Sx = {sec.Sx/1e3:.1f} × 10³ mm³
                """)
            
            with col2:
                st.markdown(f"""
                **Computed Stress Range:**
                
                f_sr = M_range / Sx
                
                f_sr = {results.M_max:.2f} × 10⁶ / {sec.Sx/1e3:.1f} × 10³
                
                **f_sr = {fatigue['f_sr']:.2f} MPa**
                """)
            
            st.markdown("---")
            
            # Allowable Stress Range
            st.markdown("### 3️⃣ Allowable Stress Range")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **From AISC Eq. A-3-1:**
                
                F_SR = (Cf / N)^(1/3) ≥ FTH
                
                F_SR = ({FATIGUE_CATEGORIES[fatigue_cat]['Cf']:.2e} / {N_cycles:,})^(1/3)
                
                F_SR = **{(FATIGUE_CATEGORIES[fatigue_cat]['Cf'] / N_cycles)**(1/3):.2f} MPa**
                
                FTH = {FATIGUE_CATEGORIES[fatigue_cat]['FTH']} MPa (threshold)
                """)
            
            with col2:
                calc_Fsr = (FATIGUE_CATEGORIES[fatigue_cat]['Cf'] / N_cycles)**(1/3)
                if calc_Fsr >= FATIGUE_CATEGORIES[fatigue_cat]['FTH']:
                    st.markdown(f"""
                    **Governing Allowable:**
                    
                    Since F_SR = {calc_Fsr:.2f} MPa ≥ FTH = {FATIGUE_CATEGORIES[fatigue_cat]['FTH']} MPa
                    
                    **Use F_SR = {fatigue['F_sr']:.2f} MPa**
                    """)
                else:
                    st.markdown(f"""
                    **Governing Allowable:**
                    
                    Since calculated F_SR < FTH
                    
                    **Use F_SR = FTH = {fatigue['F_sr']:.2f} MPa**
                    """)
            
            st.markdown("---")
            
            # Fatigue Check
            st.markdown("### 4️⃣ Fatigue Check")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                | Parameter | Value |
                |-----------|-------|
                | f_sr (stress range) | {fatigue['f_sr']:.2f} MPa |
                | F_SR (allowable) | {fatigue['F_sr']:.2f} MPa |
                | **Ratio** | **{fatigue['ratio']:.3f}** |
                """)
            
            with col2:
                if fatigue['ratio'] <= 1.0:
                    st.success(f"""
                    ✅ **FATIGUE CHECK: OK**
                    
                    f_sr = {fatigue['f_sr']:.2f} MPa ≤ F_SR = {fatigue['F_sr']:.2f} MPa
                    
                    Ratio = {fatigue['ratio']:.3f} ≤ 1.0
                    """)
                else:
                    st.error(f"""
                    ❌ **FATIGUE CHECK: FAILS**
                    
                    f_sr = {fatigue['f_sr']:.2f} MPa > F_SR = {fatigue['F_sr']:.2f} MPa
                    
                    Ratio = {fatigue['ratio']:.3f} > 1.0
                    
                    **Recommendations:**
                    - Increase section size to reduce stress range
                    - Use better weld details (higher category)
                    - Consider full penetration welds
                    """)
            
            st.markdown("---")
            
            # Fatigue Life Plot
            st.markdown("### 📈 S-N Curve Visualization")
            
            # Create S-N curve plot
            fig_sn = go.Figure()
            
            cycles = np.logspace(4, 8, 100)
            for cat_name, cat_data in [('B', FATIGUE_CATEGORIES['B']), ('C', FATIGUE_CATEGORIES['C']), 
                                       ('D', FATIGUE_CATEGORIES['D']), ('E', FATIGUE_CATEGORIES['E'])]:
                F_sr_curve = np.maximum((cat_data['Cf'] / cycles)**(1/3), cat_data['FTH'])
                fig_sn.add_trace(go.Scatter(x=cycles, y=F_sr_curve, mode='lines', name=f"Cat. {cat_name}"))
            
            # Add design point
            fig_sn.add_trace(go.Scatter(
                x=[N_cycles], y=[fatigue['f_sr']], mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name=f'Design Point (Cat. {fatigue_cat})'
            ))
            
            fig_sn.update_layout(
                title='AISC Fatigue Categories - S-N Curves',
                xaxis=dict(title='Number of Cycles (N)', type='log'),
                yaxis=dict(title='Stress Range (MPa)', range=[0, 200]),
                height=400, showlegend=True
            )
            st.plotly_chart(fig_sn, use_container_width=True)
    
    # TAB 5: STIFFENERS
    with tabs[4]:
        st.subheader("Stiffener Design")
        if not (stiff.has_transverse or stiff.has_bearing or stiff.has_longitudinal):
            st.info("No stiffeners specified.")
        
        if stiff.has_transverse and trans_check['required']:
            with st.expander("Transverse Stiffeners", expanded=True):
                for chk in trans_check['checks']:
                    st.write(f"{'✅' if chk['ok'] else '❌'} {chk['name']}: {chk['demand']} vs {chk['capacity']}")
        
        if stiff.has_bearing and bearing_check['required']:
            with st.expander("Bearing Stiffeners", expanded=True):
                for chk in bearing_check['checks']:
                    st.write(f"{'✅' if chk['ok'] else '❌'} {chk['name']}: {chk['demand']} vs {chk['capacity']}")
    
    # TAB 6: WELD DESIGN (conditional)
    tab_offset = 0
    if weld_check is not None:
        tab_offset = 1
        with tabs[5]:
            st.subheader("🔥 Weld Design for Built-up Section")
            st.markdown("**Per AISC 360-16 Chapter J2 and AWS D1.1**")
            
            # Weld info
            col1, col2, col3 = st.columns(3)
            col1.metric("Weld Type", WELD_TYPES[weld_check['weld_type']]['name'])
            col2.metric("Electrode", f"{weld_check['electrode']} ({weld_check['FEXX']} MPa)")
            col3.metric("Weld Size", f"{weld_check['weld_size']:.1f} mm")
            
            # Status
            if weld_check['ok']:
                st.success(f"✅ **WELD DESIGN: PASS** - Ratio = {weld_check['ratio']:.3f}")
            else:
                st.error(f"❌ **WELD DESIGN: FAIL** - Ratio = {weld_check['ratio']:.3f}")
            
            st.markdown("---")
            
            # ============ DETAILED SHEAR FLOW CALCULATION ============
            st.markdown("### 📐 Shear Flow Calculation")
            st.markdown("""
            For built-up I-sections, the flange-to-web welds must resist the horizontal shear flow 
            that develops due to bending. The shear flow is maximum at the neutral axis and 
            transfers shear between the flange and web.
            """)
            
            # Section properties for calculation
            tf = sec.tf_top if sec.tf_top > 0 else sec.tf
            bf = sec.bf_top if sec.bf_top > 0 else sec.bf
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Step 1: Section Properties")
                st.markdown(f"""
                **Flange dimensions:**
                - Width: b_f = {bf:.0f} mm
                - Thickness: t_f = {tf:.1f} mm
                - Area: A_f = b_f × t_f = {bf:.0f} × {tf:.1f} = **{weld_check['A_flange']:.0f} mm²**
                
                **Distance from N.A. to flange centroid:**
                - ȳ = (d - t_f) / 2 = ({sec.d:.0f} - {tf:.1f}) / 2 = **{weld_check['y_flange']:.1f} mm**
                
                **First moment of area (Q):**
                - Q = A_f × ȳ = {weld_check['A_flange']:.0f} × {weld_check['y_flange']:.1f}
                - **Q = {weld_check['Q']:.0f} mm³** = **{weld_check['Q']/1e6:.4f} × 10⁶ mm³**
                
                **Moment of inertia:**
                - I_x = **{sec.Ix/1e6:.2f} × 10⁶ mm⁴**
                """)
            
            with col2:
                st.markdown("#### Step 2: Shear Flow Formula")
                st.latex(r"q = \frac{V \times Q}{I}")
                
                st.markdown(f"""
                **Where:**
                - V = Design shear force = {V_design:.2f} kN = **{V_design * 1000:.0f} N**
                - Q = First moment of area = {weld_check['Q']:.0f} mm³
                - I = Moment of inertia = {sec.Ix:.0f} mm⁴
                
                **Calculation:**
                """)
                st.latex(rf"q = \frac{{{V_design * 1000:.0f} \times {weld_check['Q']:.0f}}}{{{sec.Ix:.0f}}}")
                st.markdown(f"**q = {weld_check['shear_flow']:.2f} N/mm** (total shear flow at flange-web junction)")
            
            st.markdown("---")
            
            st.markdown("#### Step 3: Shear Flow per Weld Line")
            st.markdown(f"""
            With **{weld_data.num_welds} weld lines** (welds on both sides of web):
            
            **q per weld = q / n = {weld_check['shear_flow']:.2f} / {weld_data.num_welds} = {weld_check['q_per_weld']:.2f} N/mm**
            """)
            
            # Visual representation
            st.markdown("#### Weld Location Diagram")
            
            # Create a simple cross-section diagram showing weld locations
            fig_weld = go.Figure()
            
            # Draw I-section cross-section
            hw = sec.hw if sec.hw > 0 else sec.d - 2 * tf
            bf_bot = sec.bf_bot if sec.bf_bot > 0 else bf
            tf_bot = sec.tf_bot if sec.tf_bot > 0 else tf
            
            # Scale for drawing
            scale = 1
            
            # Top flange
            fig_weld.add_shape(type="rect", x0=-bf/2, y0=hw/2, x1=bf/2, y1=hw/2+tf,
                             fillcolor="lightsteelblue", line=dict(color="navy", width=2))
            # Bottom flange
            fig_weld.add_shape(type="rect", x0=-bf_bot/2, y0=-hw/2-tf_bot, x1=bf_bot/2, y1=-hw/2,
                             fillcolor="lightsteelblue", line=dict(color="navy", width=2))
            # Web
            fig_weld.add_shape(type="rect", x0=-sec.tw/2, y0=-hw/2, x1=sec.tw/2, y1=hw/2,
                             fillcolor="lightblue", line=dict(color="navy", width=2))
            
            # Weld locations (red triangles at flange-web junction)
            weld_size_draw = weld_check['weld_size'] * 3  # Exaggerated for visibility
            
            # Top flange welds
            fig_weld.add_trace(go.Scatter(
                x=[-sec.tw/2 - weld_size_draw/2, -sec.tw/2, -sec.tw/2 - weld_size_draw/2],
                y=[hw/2, hw/2, hw/2 + weld_size_draw/2],
                fill="toself", fillcolor="red", line=dict(color="darkred", width=1),
                name="Fillet Weld", showlegend=True
            ))
            fig_weld.add_trace(go.Scatter(
                x=[sec.tw/2 + weld_size_draw/2, sec.tw/2, sec.tw/2 + weld_size_draw/2],
                y=[hw/2, hw/2, hw/2 + weld_size_draw/2],
                fill="toself", fillcolor="red", line=dict(color="darkred", width=1),
                showlegend=False
            ))
            
            # Bottom flange welds
            fig_weld.add_trace(go.Scatter(
                x=[-sec.tw/2 - weld_size_draw/2, -sec.tw/2, -sec.tw/2 - weld_size_draw/2],
                y=[-hw/2, -hw/2, -hw/2 - weld_size_draw/2],
                fill="toself", fillcolor="red", line=dict(color="darkred", width=1),
                showlegend=False
            ))
            fig_weld.add_trace(go.Scatter(
                x=[sec.tw/2 + weld_size_draw/2, sec.tw/2, sec.tw/2 + weld_size_draw/2],
                y=[-hw/2, -hw/2, -hw/2 - weld_size_draw/2],
                fill="toself", fillcolor="red", line=dict(color="darkred", width=1),
                showlegend=False
            ))
            
            # Neutral axis
            fig_weld.add_shape(type="line", x0=-bf/2-20, y0=0, x1=bf/2+20, y1=0,
                             line=dict(color="green", width=2, dash="dash"))
            fig_weld.add_annotation(x=bf/2+30, y=0, text="N.A.", showarrow=False, font=dict(color="green"))
            
            # Dimension annotations
            fig_weld.add_annotation(x=0, y=hw/2+tf+30, text=f"b_f = {bf:.0f} mm", showarrow=False)
            fig_weld.add_annotation(x=-bf/2-40, y=0, text=f"h_w = {hw:.0f} mm", showarrow=False, textangle=-90)
            
            # Shear flow arrows
            fig_weld.add_annotation(x=-sec.tw/2-50, y=hw/2, ax=-sec.tw/2-50, ay=hw/2-60,
                                   showarrow=True, arrowhead=2, arrowcolor="orange", arrowwidth=3)
            fig_weld.add_annotation(x=-sec.tw/2-70, y=hw/2-30, text="q", font=dict(color="orange", size=14), showarrow=False)
            
            fig_weld.update_layout(
                title="Cross-Section with Weld Locations",
                xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400, showlegend=True,
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig_weld, use_container_width=True)
            
            st.markdown("---")
            
            # Check Results Table
            st.markdown("### Design Checks Summary")
            
            check_data = []
            for chk in weld_check['checks']:
                status = "✅ PASS" if chk['ok'] else "❌ FAIL"
                ratio_str = f"{chk.get('ratio', '-'):.3f}" if 'ratio' in chk else "-"
                check_data.append([
                    chk['name'],
                    chk['demand'],
                    chk['capacity'],
                    ratio_str,
                    status,
                    chk.get('reference', '')
                ])
            
            df_checks = pd.DataFrame(check_data, columns=['Check', 'Demand', 'Capacity', 'Ratio', 'Status', 'Reference'])
            st.dataframe(df_checks, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Weld Strength Details
            if 'weld_strength' in weld_check:
                ws = weld_check['weld_strength']
                st.markdown("### Weld Strength Calculation")
                
                if weld_check['weld_type'] == 'fillet':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        **Fillet Weld Geometry:**
                        - Leg size: w = {weld_check['weld_size']:.1f} mm
                        - Effective throat: a = 0.707 × w
                        - a = 0.707 × {weld_check['weld_size']:.1f} = **{ws['throat']:.2f} mm**
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Weld Strength (AISC J2.4):**
                        - F_nw = 0.60 × F_EXX = 0.60 × {weld_check['FEXX']} = **{ws['Fnw']:.1f} MPa**
                        - R_n = F_nw × a = {ws['Fnw']:.1f} × {ws['throat']:.2f} = **{ws['Rn']:.2f} N/mm**
                        - Ω = {ws['omega']:.2f} (ASD safety factor)
                        - **R_n/Ω = {ws['Rn_allow']:.2f} N/mm**
                        """)
                    
                    st.info(f"**Governing failure mode:** {ws['governs']}")
                else:
                    st.markdown(f"""
                    **Groove Weld Strength (AISC J2.3):**
                    
                    - Effective throat: **{ws['throat']:.2f} mm**
                    - Shear strength: R_n = **{ws['Rn_shear']:.2f} N/mm**
                    - Allowable: R_n/Ω = **{ws['Rn_allow_shear']:.2f} N/mm**
                    """)
            
            # Recommended weld size
            if 'w_required' in weld_check:
                st.markdown("---")
                st.markdown("### Weld Size Summary")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Minimum (Table J2.4)", f"{weld_check['w_min']:.1f} mm")
                col2.metric("Required (by calculation)", f"{weld_check['w_required']:.1f} mm")
                col3.metric("Maximum (J2.2b)", f"{weld_check['w_max']:.1f} mm")
                
                if weld_check['weld_size'] >= weld_check['w_required']:
                    st.success(f"✅ Provided {weld_check['weld_size']:.1f} mm ≥ Required {weld_check['w_required']:.1f} mm - **ADEQUATE**")
                else:
                    st.error(f"❌ Provided {weld_check['weld_size']:.1f} mm < Required {weld_check['w_required']:.1f} mm - **INCREASE WELD SIZE!**")
    
    # TAB: BRACKET LOADS
    with tabs[5 + tab_offset]:
        st.subheader("Bracket Design Loads")
        
        total_lat = sum(c.get_lateral_load_per_wheel() * c.num_wheels for c in cranes)
        max_long = max(c.get_longitudinal_force() for c in cranes)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔵 LEFT SUPPORT (A)")
            st.metric("V_max (Vertical)", f"{results.R_A_max + R_self:.1f} kN")
            st.metric("V_min (Vertical)", f"{results.R_A_min + R_self:.1f} kN")
            st.metric("H (Lateral)", f"{total_lat:.1f} kN")
            st.metric("L (Longitudinal)", f"{max_long:.1f} kN")
        with col2:
            st.markdown("### 🔴 RIGHT SUPPORT (B)")
            st.metric("V_max (Vertical)", f"{results.R_B_max + R_self:.1f} kN")
            st.metric("V_min (Vertical)", f"{results.R_B_min + R_self:.1f} kN")
            st.metric("H (Lateral)", f"{total_lat:.1f} kN")
            st.metric("L (Longitudinal)", f"{max_long:.1f} kN")
        
        st.markdown("---")
        st.markdown("**Notes:**")
        st.write("• V includes beam self-weight")
        st.write("• H = lateral thrust at top of rail")
        st.write("• L = longitudinal braking force")
    
    # TAB: REPORT
    with tabs[6 + tab_offset]:
        st.subheader("📄 Design Report")
        
        # Generate report content
        report = f"""# CRANE RUNWAY BEAM DESIGN REPORT
## Per AISC 360-16 (ASD), Design Guide 7, CMAA 70/74

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. PROJECT DATA

### 1.1 Beam Geometry
- Span: L = {beam_span:.2f} m
- Unbraced Length: Lb = {Lb:.2f} m

### 1.2 Material: {steel_grade}
- Yield Strength: Fy = {Fy} MPa
- Ultimate Strength: Fu = {Fu} MPa

### 1.3 Section: {sec.name}{' + ' + sec.cap_name if sec.has_cap else ''}
- Depth: d = {sec.d:.0f} mm
- Flange: bf = {max(sec.bf_top, sec.bf_bot) if sec.bf_top > 0 else sec.bf:.0f} mm, tf = {sec.tf_top if sec.tf_top > 0 else sec.tf:.1f} mm
- Web: tw = {sec.tw:.1f} mm, hw = {sec.hw:.0f} mm
- Area: A = {sec.A:.0f} mm²
- Ix = {sec.Ix/1e6:.3f} ×10⁶ mm⁴
- Sx = {sec.Sx/1e3:.1f} ×10³ mm³
- Mass = {sec.mass:.1f} kg/m

---

## 2. CRANE LOADING

### 2.1 Crane Data
"""
        for i, crane in enumerate(cranes):
            P_max, _ = crane.calc_wheel_loads()
            report += f"""
**Crane {i+1}:**
- Capacity: {crane.capacity_tonnes:.1f} T, Wheel Base: {crane.wheel_base:.2f} m
- Max Wheel Load: {P_max:.2f} kN (no impact)
- Max Wheel Load with Impact: {crane.get_wheel_load_with_impact():.2f} kN
"""
        
        report += f"""
### 2.2 Design Forces
- M_crane = {results.M_max:.2f} kN-m at x = {results.M_max_location:.2f} m
- M_self = {M_self:.2f} kN-m
- **M_design = {M_design:.2f} kN-m**
- V_crane = {results.V_max:.2f} kN
- **V_design = {V_design:.2f} kN**

---

## 3. SECTION CLASSIFICATION (Table B4.1b)

### Flange: {cmp['flange_class']}
- λf = {cmp['lambda_f']:.2f}, λpf = {cmp['lambda_pf']:.2f}, λrf = {cmp['lambda_rf']:.2f}

### Web: {cmp['web_class']}
- λw = {cmp['lambda_w']:.2f}, λpw = {cmp['lambda_pw']:.2f}, λrw = {cmp['lambda_rw']:.2f}

---

## 4. FLEXURAL STRENGTH (Chapter F)

- Mp = {flex['Mp']:.2f} kN-m
- Lp = {flex['Lp']/1000:.3f} m, Lr = {flex['Lr']/1000:.3f} m
- Limit State: {flex['limit_state']}
- Mn = {flex['Mn']:.2f} kN-m
- **Mn/Ω = {flex['Mn_allow']:.2f} kN-m**
- **Ratio = {flex_ratio:.3f}** {'✓ OK' if flex_ratio <= 1.0 else '✗ NG'}

---

## 5. SHEAR STRENGTH (Chapter G)

- Aw = {shear['Aw']:.0f} mm², h/tw = {shear['h_tw']:.1f}
- kv = {shear['kv']:.2f}, Cv1 = {shear['Cv1']:.3f}
- Limit State: {shear['limit_state']}
- Vn = {shear['Vn']:.2f} kN
- **Vn/Ω = {shear['Vn_allow']:.2f} kN**
- **Ratio = {shear_ratio:.3f}** {'✓ OK' if shear_ratio <= 1.0 else '✗ NG'}

---

## 6. WEB LOCAL EFFECTS (J10)

### Web Local Yielding (J10.2)
- k = {wly['k']:.1f} mm, lb = {lb_bearing:.0f} mm
- Rn = {wly['Rn']:.2f} kN, **Rn/Ω = {wly['Rn_allow']:.2f} kN**
- **Ratio = {wly['ratio']:.3f}** {wly['status']}

### Web Crippling (J10.3)
- Rn = {wcr['Rn']:.2f} kN, **Rn/Ω = {wcr['Rn_allow']:.2f} kN**
- **Ratio = {wcr['ratio']:.3f}** {wcr['status']}

---

## 7. DEFLECTION CHECK

- Crane Class: {crane_class} - {CRANE_CLASSES[crane_class]['name']}
- Limit: L/{CRANE_CLASSES[crane_class]['defl_limit']} = {delta_limit:.2f} mm
- δ = {delta:.2f} mm
- **Ratio = {defl_ratio:.3f}** {'✓ OK' if defl_ratio <= 1.0 else '✗ NG'}

---
"""
        
        if check_fatigue_enabled:
            report += f"""
## 8. FATIGUE CHECK (Appendix 3)

- Category: {fatigue_cat}, N = {N_cycles:,} cycles
- f_sr = {fatigue['f_sr']:.2f} MPa
- F_sr = {fatigue['F_sr']:.2f} MPa (FTH = {fatigue['FTH']:.1f} MPa)
- **Ratio = {fatigue['ratio']:.3f}** {fatigue['status']}

---
"""
        
        report += f"""
## DESIGN SUMMARY

| Check | Demand | Capacity | Ratio | Status |
|-------|--------|----------|-------|--------|
| Flexure | {M_design:.1f} kN-m | {flex['Mn_allow']:.1f} kN-m | {flex_ratio:.3f} | {'✓' if flex_ratio <= 1.0 else '✗'} |
| Shear | {V_design:.1f} kN | {shear['Vn_allow']:.1f} kN | {shear_ratio:.3f} | {'✓' if shear_ratio <= 1.0 else '✗'} |
| Web Yielding | {results.R_A_max:.1f} kN | {wly['Rn_allow']:.1f} kN | {wly['ratio']:.3f} | {wly['status']} |
| Web Crippling | {results.R_A_max:.1f} kN | {wcr['Rn_allow']:.1f} kN | {wcr['ratio']:.3f} | {wcr['status']} |
| Deflection | {delta:.2f} mm | {delta_limit:.1f} mm | {defl_ratio:.3f} | {'✓' if defl_ratio <= 1.0 else '✗'} |
"""
        if check_fatigue_enabled:
            report += f"| Fatigue | {fatigue['f_sr']:.1f} MPa | {fatigue['F_sr']:.1f} MPa | {fatigue['ratio']:.3f} | {fatigue['status']} |\n"
        
        report += f"""
**Overall: {'✓ ALL CHECKS PASS' if all_ok else '✗ SOME CHECKS FAIL'}**

---

## BRACKET DESIGN LOADS

| Support | V_max (kN) | V_min (kN) | H (kN) | L (kN) |
|---------|------------|------------|--------|--------|
| Left (A) | {results.R_A_max + R_self:.1f} | {results.R_A_min + R_self:.1f} | {total_lat:.1f} | {max_long:.1f} |
| Right (B) | {results.R_B_max + R_self:.1f} | {results.R_B_min + R_self:.1f} | {total_lat:.1f} | {max_long:.1f} |

---

*Report generated by Crane Runway Beam Design Pro V6.0*
"""
        
        # Report download section
        st.markdown("### 📥 Download Reports")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            st.download_button(
                label="📄 Download Markdown Report",
                data=report,
                file_name=f"CraneRunway_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col_dl2:
            # PDF Report Generation
            if REPORTLAB_AVAILABLE:
                # Project info inputs
                with st.expander("📝 Project Information (Optional)", expanded=False):
                    proj_col1, proj_col2 = st.columns(2)
                    with proj_col1:
                        project_name = st.text_input("Project Name", value="Crane Runway Beam Design")
                        designer = st.text_input("Designer", value="")
                    with proj_col2:
                        project_number = st.text_input("Project Number", value="")
                        checker = st.text_input("Checker", value="")
                
                if st.button("📕 Generate Academic PDF Report", use_container_width=True, type="primary"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            pdf_data = generate_academic_pdf_report(
                                beam_span=beam_span, Lb=Lb, steel_grade=steel_grade, 
                                Fy=Fy, Fu=Fu, sec=sec, cranes=cranes, 
                                crane_class=crane_class, N_cycles=N_cycles, fatigue_cat=fatigue_cat,
                                results=results, M_self=M_self, V_self=V_self, 
                                M_design=M_design, V_design=V_design,
                                cmp=cmp, flex=flex, shear=shear, wly=wly, wcr=wcr, 
                                fatigue=fatigue, delta=delta, delta_limit=delta_limit,
                                stiff=stiff, trans_check=trans_check, bearing_check=bearing_check,
                                flex_ratio=flex_ratio, shear_ratio=shear_ratio, defl_ratio=defl_ratio,
                                check_fatigue_enabled=include_fatigue_in_pdf,
                                weld_check=weld_check,
                                project_name=project_name if 'project_name' in dir() else "Crane Runway Design",
                                project_number=project_number if 'project_number' in dir() else "",
                                designer=designer if 'designer' in dir() else "",
                                checker=checker if 'checker' in dir() else ""
                            )
                            
                            if pdf_data:
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_data,
                                    file_name=f"CraneRunway_AcademicReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                                st.success("✅ PDF report generated successfully!")
                            else:
                                st.error("Failed to generate PDF report")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
            else:
                st.warning("📦 Install reportlab for PDF generation: `pip install reportlab`")
        
        st.markdown("---")
        st.markdown("### Report Preview")
        st.markdown(report)
    
    st.markdown("---")
    st.caption("Crane Runway Beam Design Pro V6.0 | AISC 360-16 (ASD) | Design Guide 7 | CMAA 70/74")


if __name__ == "__main__":
    main()
