# Vehicle routing simulation 

# Overview
This code is developed to optimize the truck allocation process for a trucking logistics company. It focuses on minimizing transportation costs by intelligently assigning trucks to various shipment locations based on the weight of the goods and the distance between hubs. The code utilizes the Clark-Wright Savings Algorithm along with other optimization techniques to achieve cost-effective routing solutions.

# Features
The code begins by organizing a list of available trucks in descending order of their capacity. It calculates the maximum truck capacity to aid in efficient allocation logic.

Implements a function (Distance_calculator) to compute the geographical distance between two points. This function is essential for determining the distances between various hubs. The distances are calculated using the Haversine formula, considering the Earth's curvature.

The core of the script uses the Clark-Wright Savings Algorithm for truck assignment. This algorithm calculates savings in distance and cost by combining trips and comparing them against a set threshold.


# Simulation and Results Analysis
Runs a simulation over a specified planning horizon to assign trucks to shipments on different days.
Outputs the results including the total cost with and without vehicle routing, the number of trucks used, and estimated cost savings.
Visualizes the data through histograms and bar charts for better insight into weight distribution and truck utilization.

# Output
The script outputs the total costs associated with truck allocations both with and without vehicle routing optimization.
It provides a detailed breakdown of truck types used and the number of trucks required for each type.
Visual representations of the data are provided in the form of histograms and bar charts.
