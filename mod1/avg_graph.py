import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Frame": list(range(29)),  # Frame numbers from 0 to 28
    "YOLO Count": [82, 73, 75, 65, 56, 71, 67, 71, 60, 74, 74, 74, 64, 73, 75, 85, 79, 71, 90, 79, 77, 76, 90, 82, 85, 73, 76, 74, 71],
    "RCNN Count": [100, 100, 93, 100, 100, 96, 95, 100, 99, 97, 94, 86, 98, 100, 94, 100, 98, 100, 100, 100, 93, 100, 100, 99, 100, 100, 100, 100, 100]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate average of YOLO and RCNN counts
df['Average Count'] = (df['YOLO Count'] + df['RCNN Count']) / 2

# Plotting
plt.figure(figsize=(10, 5))
# plt.plot(df['Frame'], df['YOLO Count'], marker='o', label='YOLO Count', color='blue')
# plt.plot(df['Frame'], df['RCNN Count'], marker='o', label='RCNN Count', color='orange')
plt.plot(df['Frame'], df['Average Count'], marker='o', label='Average Count', color='green')

# Adding titles and labels
plt.title('Average Count per Frame')
plt.xlabel('Frame')
plt.ylabel('Count')
plt.xticks(df['Frame'])  # Set x-ticks to frame numbers
plt.legend()
plt.grid()

# Show the plot
plt.show()