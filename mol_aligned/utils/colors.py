import matplotlib as mpl

IBM_COLORS = ["#648fff", "#ffb000", "#dc267f", "#785ef0", "#fe6100", "#000000", "#ffffff"]
COLORS = IBM_COLORS

# set color cycle for matplotlib
def set_mpl_color_cycle():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)

# function to reset to default matplotlib color cycle
def reset_mpl_color_cycle():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color'])

set_mpl_color_cycle()