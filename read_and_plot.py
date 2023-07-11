import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy import float64

# parse tsp file
def read(filename):
    with open(filename) as fp:
        text = fp.readlines()

        for x, line in enumerate(text):
            if line.startswith('DIMENSION :'):
                dimension = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                coord_line = x
                break

        fp.seek(0,0)

        return pd.read_csv(
            fp,
            skiprows=coord_line+1,
            sep=' ',
            names=['city','y','x'],
            dtype={'city':str, 'x':float64, 'y':float64},
            header=None,
            nrows=dimension
        )

# plotting neural network or final path
def plot(weights, nodes, name, path=None):
    rcParams['agg.path.chunksize'] = 100000
    fig = plt.figure(figsize=(6,6), frameon=True)
    axis = fig.add_axes([0,0,1,1])
    axis.set_aspect('equal')
    axis.axis('off')

    x_range = nodes['x'].max() - nodes['x'].min()
    y_range = nodes['y'].max() - nodes['y'].min()
    axis.fill_between([nodes['x'].min() - 0.1 * x_range, nodes['x'].max() + 0.1 * x_range],
                      [nodes['y'].min() - 0.1 * y_range, nodes['y'].min() - 0.1 * y_range],
                      [nodes['y'].max() + 0.1 * y_range, nodes['y'].max() + 0.1 * y_range],
                      color='white')
    
    if path is not None:
        axis.scatter(nodes['x'], nodes['y'], color='black', s=4)
        path = nodes.reindex(path)
        path.loc[path.shape[0]] = path.iloc[0]
        path.plot(x='x', y='y', color='red', linewidth=1, ax=axis)
        name = 'output/path.png'
        axis.legend().remove()
    else:
        axis.scatter(nodes['x'], nodes['y'], color='red', s=2)
        axis.plot(weights[:, 0], weights[:, 1], color='black', ls='-', markersize=0.01)

    axis.set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
    
    plt.close()