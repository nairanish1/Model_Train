import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data CSV
anthropomorphic_data = pd.read_csv('/Users/anishnair/Global_HRTF_VAE/HRTF_Data/Antrhopometric-measures/AntrhopometricMeasures.csv')

def anthropometric_data_preprocessing(anthropomorphic_data):

    # Correct indices (0-based indexing)
    subjects_to_remove = [17, 78, 91]
    anthropomorphic_data = anthropomorphic_data.drop(index=subjects_to_remove).reset_index(drop=True)

    # Remove the first column
    anthropomorphic_data = anthropomorphic_data.iloc[:, 1:]

    # Convert the DataFrame to a NumPy array for processing
    measurements = anthropomorphic_data.values
    
    # Check if NaNs exist after removal, handle if necessary
    if np.isnan(measurements).any():
        measurements = np.nan_to_num(measurements, nan=np.nanmean(measurements))

    # Calculate the mean and standard deviations for each feature
    means = np.mean(measurements, axis=0)
    stds = np.std(measurements, axis=0)

    # Ensure standard deviations are not zero
    stds[stds == 0] = 1

    # Normalize measurements using sigmoid-like transformation
    normalized_measurements = 1 / (1 + np.exp(-(measurements - means) / stds))

    # Convert back to DataFrame
    normalized_measurements_df = pd.DataFrame(normalized_measurements, columns=anthropomorphic_data.columns)

    return normalized_measurements_df

# Preprocess the anthropometric data
normalized_data = anthropometric_data_preprocessing(anthropomorphic_data)

# Save normalized data into a new CSV file
normalized_csv_data = '/Users/anishnair/Global_HRTF_VAE/Normalized_Anthropometric_Data.csv'
normalized_data.to_csv(normalized_csv_data, index=False)

# Print the first few rows of the normalized data for verification
print("First few rows of the normalized data:\n", normalized_data.head())

# Plot distribution of normalized data
plt.figure(figsize=(12,6))
plt.hist(normalized_data.values.flatten(), bins=30, density=True, alpha=0.6, color='g')
plt.title("Distribution of the Normalized Anthropometric Data")
plt.xlabel("Normalized Values")
plt.ylabel("Density")
plt.grid()
plt.show()
