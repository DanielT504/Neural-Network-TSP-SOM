from PIL import Image
import sys
import numpy as np
import read_and_plot as rp
import re
import os

# self-organizing map
def map(tsp_unordered, alpha, pop_ratio, alpha_decay, radius_decay):
    # standardize all coordinates to a range and domain of [0,1]
    tsp_ordered = tsp_unordered.copy()
    x_range = tsp_ordered['x'].max() - tsp_ordered['x'].min()
    y_range = tsp_ordered['y'].max() - tsp_ordered['y'].min()
    ratio = x_range / y_range, 1
    ratio = np.array(ratio) / max(ratio)
    norm = (tsp_ordered[['x','y']]
            - tsp_ordered[['x','y']].min()) / (tsp_ordered[['x','y']].max()
                                                - tsp_ordered[['x','y']].min())
    tsp_ordered[['x', 'y']] = norm * ratio

    # random neuron population relative to problem size
    n = pop_ratio*tsp_ordered.shape[0]
    nn = np.random.random_sample((n, 2))
    print('Neural network population: {}'.format(n))

    for it in range(100000):
        print(' Iteration {} out of 100,000'.format(it), end="\r")

        # select a winner neuron by euclidean dstance to a given city
        node = tsp_ordered[['x','y']].values[np.random.choice(len(tsp_ordered))]
        weights = range_gaussian(nn.shape[0], n,
                                 np.linalg.norm(nn - node,
                                                axis=1).argmin())
        nn += weights[:,np.newaxis] * alpha * (node - nn)
        
        if not it % 1000 and it != 0:
            rp.plot(nn, tsp_ordered, 'output/{}k.png'.format(int(it/1000)))

        n *= radius_decay
        alpha *= alpha_decay
        if n < 1 or alpha < 0.01:
            print('Radius or learning rate reached maximum decay',
                'at iteration {}'.format(it))
            
            if it != 100000:
                rp.plot(nn, tsp_ordered, 'output/{}k.png'.format(int(it/1000)))
            break
    else:
        rp.plot(nn, tsp_ordered, 'output/{}k.png'.format(int((it + 1)/1000)))

    # sort nodes according to the network route
    soln_path = sort_nodes(nn, tsp_ordered)
    rp.plot(None, tsp_ordered, '', soln_path)
    return soln_path

# find range gaussian distribution around a center index in a circular network.
def range_gaussian(domain, width, center):
    # circular distance from each point to the center
    deltas = center - np.arange(domain)
    if np.any(center < np.arange(domain)):
        deltas = np.where(center < np.arange(domain), -deltas, deltas)

    distances = deltas.copy()
    cond = deltas > domain / 2
    if np.any(cond):
        distances[cond] = domain - deltas[cond]
    distances **= 2
    
    # gaussian distribution around center
    if (width//10) < 1:
        return np.exp(-(distances)/2)
    else:
        return np.exp(-(distances)/(2*((width//10)**2)))

# sort neurons to find a route
def sort_nodes(neural_network, nodes):
    winners = [np.linalg.norm(neural_network - n, axis=1).argmin() for n in nodes[['x','y']].values]
    sorted_indices = np.argsort(winners)
    sorted_nodes = nodes.iloc[sorted_indices].reset_index()
    return sorted_nodes['index']

# numerate output PNGs and place path at the end
def extract_numeric_value(filename):
    match = re.search(r'(\d+)k.png', filename)
    if match:
        return int(match.group(1)) * 1000
    elif filename == "path.png":
        return float('inf')
    else:
        return -1

if __name__ == '__main__':
    if len(sys.argv) == 2:
        # parse file, solve with self-organizing map, calculate distance
        tsp_unordered = rp.read(sys.argv[1])
        soln_path = map(tsp_unordered, 0.5, 5, 0.99999, 0.9999)
        tsp_unordered = tsp_unordered.reindex(soln_path)
        route_length = np.sum(np.linalg.norm(tsp_unordered[['x', 'y']]
                                             - np.roll(tsp_unordered[['x', 'y']], 1, axis=0), axis=1))
        print('Solution path: {}km              '.format(int(route_length)))

        # generating GIF
        frames = []
        img_dir = "output"
        img_files = [filename for filename in os.listdir(img_dir) if filename.endswith(".png")]
        img_files.sort(key=lambda x: extract_numeric_value(x))
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            new_frame = Image.open(img_path)
            frames.append(new_frame)

        path_index = img_files.index('path.png')
        if path_index >= 0:
            img_files.append(img_files.pop(path_index))
            frames.append(frames.pop(path_index))

        num_pause_frames = 10
        pause_frame = frames[-1]
        for _ in range(num_pause_frames):
            frames.append(pause_frame.copy())
        frames[0].save('output/process.gif', format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=1)
    else:
        print("Please use command \"python main.py <filename>.tsp\"")