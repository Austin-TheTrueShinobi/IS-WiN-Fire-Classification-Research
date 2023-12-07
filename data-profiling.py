
import pandas as pd
from PIL import Image
import os
import numpy as np
from ydata_profiling import ProfileReport

# Specify the path to the directory containing your images
image_directory = "Images"

# List all files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Create an empty DataFrame to store flattened pixel intensities
df = pd.DataFrame()

# Loop through each image file
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_directory, image_file)

    # Read the image using PIL
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Flatten the pixel intensities and add them to the DataFrame
    df[image_file] = image_array.flatten()

# Generate a profiling report for all columns
profile = ProfileReport(df, title='Image Data Profiling Report', explorative=True)

# Display the report in the notebook
profile.to_notebook_iframe()

# Save the report to an HTML file
profile.to_file("report.html")
