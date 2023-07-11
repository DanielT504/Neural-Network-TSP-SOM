# Using a Neural Network to Solve Traveling Salesman Problems (TSP)

The neural network used is called a self-organizing map (SOM). The TSP is a problem involving a list of cities with distances between them. The objective is to find the shortest route that visits each city exactly once and loops. The SOM algorithm provides an approach to find an optimized solution for the TSP by simulating a network of neurons, where each neuron represents a random point in the search space. On each iteration, the neuron closest to a randomly selected city, as well as its neighboring neurons, are moved closer to said city. At a large number of iterations, the neurons should arrange themselves in a density similar to the city distribution, approximating the optimal route. With each iteration, the learning rate and the radius of influence will decay to allow for more fine-tuned changes. The final route is determined by the closest cities to each point along the circular network. A GIF animation is produced to demonstrate the evolution of the solution over time, as well as the individual progress frames.
![example](https://github.com/DanielT504/Neural-Network-TSP-SOM/assets/62156098/3dc4c238-c93e-4e4b-aad6-2f1de2908dd3)
You can run the code using "python main.py `<filename>`.tsp".

There are sample .tsp files (the standard problem format for TSP problems) provided,
or you can try new ones from https://www.math.uwaterloo.ca/tsp/world/countries.html

NOTE: The code supports up to approximately 90 000 neurons before calculations overflow,
so the neuron to city ratio might need to be adjusted for larger problem files

This code improves upon an algorithm inspired by MIT licensed software (copyright included).
