from matplotlib.colors import LinearSegmentedColormap

# Color Selection
two_colors = ['indianred', 'slategray']

two_colors_map = LinearSegmentedColormap.from_list('two_colors', two_colors, N=2)

anomaly_map = LinearSegmentedColormap.from_list('anomaly', two_colors[::-1], N=100)

# Three colours
three_colors = ['indianred', 'slategray', 'cornflowerblue']


three_color_map = LinearSegmentedColormap.from_list('three_colors', three_colors[::-1], N=3)