import pickle

# Load data from the pickle file
with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Print the loaded data
print(loaded_data)
