import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
df = pd.read_csv('database.csv')

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to CSV files
train_df.to_csv('train_database.csv', index=False)
test_df.to_csv('test_database.csv', index=False)

print("Train and test datasets saved as train_database.csv and test_database.csv")
