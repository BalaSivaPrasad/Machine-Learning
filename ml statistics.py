from statistics import mean, median, mode,variance,stdev
import numpy as np
#Central Tendency
data = [5, 10, 15, 20, 25,20,20]
mean_value = mean(data)
median_value = median(data)
mode_value=mode(data)
print(f"Dataset: {data}")
print(f"mean_value:{mean_value}")
print(f"median_value:{median_value}")
print(f"mode_value:{mode_value}")
range_value = max(data) - min(data)
# Calculate Quartiles
q1 = np.percentile(data, 25) # First Quartile (25th percentile)
q3 = np.percentile(data, 75) # Third Quartile (75th percentile)
# Calculate Interquartile Range (IQR)
iqr = q3 - q1
#varaince and Standard Deviation
v = variance(data)
std_dev = stdev(data)
# Output Results
print(f"Dataset: {data}")
print(f"Range: {range_value}")
print(f"First Quartile (Q1): {q1}")
print(f"Third Quartile (Q3): {q3}")
print(f"Interquartile Range (IQR): {iqr}")
print(f"Variance (sigma square): {v}")
print(f"Standard Deviation (sqrt(variance)): {std_dev}")
