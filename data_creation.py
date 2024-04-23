import pandas as pd

# Read the existing dataset
df = pd.read_csv('data_copy.csv')

# Define the conditions
condition_N = df['N'] < 80
condition_P = df['P'] < 50
condition_K = df['K'] < 30
condition_Temperature = df['Temperature'] > 32
condition_Humidity = df['Humidity'] > 75
condition_PH = df['PH'] > 8
condition_Water_content = df['Water_content'] > 700
condition_LDR = df['LDR'] < 200
condition_CO2 = df['CO2'] < 1000

# Set 'Diseased' column based on conditions
df['Diseased'] = (condition_N | condition_P | condition_K | condition_Temperature |
                  condition_Humidity | condition_PH | condition_Water_content |
                  condition_LDR | condition_CO2).astype(int)

# Write the updated dataset to a new file
df.to_csv('updated_dataset.csv', index=False)

# Display the modified dataframe
print(df)
