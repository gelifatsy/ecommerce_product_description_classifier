import os
import pandas as pd
import numpy as np

import pandas as pd

# Define categories and product attributes
categories = ['Electronics', 'Fashion', 'Home', 'Sports']
attributes = ['name', 'description', 'price', 'category']

# Generate dummy data
data = []
for i in range(100):
    category = categories[i % len(categories)]
    name = f"Product {i}"
    description = f"This is a {category} product"
    price = i * 10
    data.append([name, description, price, category])

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=attributes)


# Specify the directory to save the file
output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'generated')

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Save the dummy data to a CSV file
csv_file_path = os.path.join(output_directory, 'dummy_product_data.csv')
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved to: {csv_file_path}")
