"""
scgtac_palette.py — SOFTENED RdYlBu family (blue->cream->coral), low-saturation endpoints.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_DIV = ['#5B8DBF','#8FB7D8','#C5DEEC','#EBF1DB','#FBEFC2','#F8CB9E','#EE9F83','#DC7A68','#CF6A5B']
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list('ref_soft', _DIV)
HEATMAP_CMAP_CUSTOM = HEATMAP_CMAP

COOL    = '#5271AE'
COOL_L  = '#82C9FF'
NEUTRAL = '#AEB0B5'
WARM_L  = '#FB8275'
WARM    = '#C0584F'
METRIC_COLORS = ['#5271AE', '#DAA81C', '#C0584F']

def series_colors(n):
    return [HEATMAP_CMAP(i/max(n-1,1)) for i in range(n)]

EXPR_CMAP = mcolors.LinearSegmentedColormap.from_list('ref_warm_soft',
    ['#FDF7EA','#FBE6C2','#F5C196','#E89A7E','#D87567'])

CELLTYPE_COLORS = ['#5271AE','#CF6A5B','#82C9FF','#F79015','#DAA81C','#8FA98C',
                   '#9A7DB0','#9A9A92','#7FA6A6','#FB8275','#88A06E','#7DACD1']

def rank_colors(n=7, best='warm'):
    if best=='warm':
        return [HEATMAP_CMAP(1 - i/(n-1)) for i in range(n)]
    return [HEATMAP_CMAP(i/(n-1)) for i in range(n)]

def apply_nature_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,
        'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'legend.frameon':False,
        'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
        'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42,
    })
