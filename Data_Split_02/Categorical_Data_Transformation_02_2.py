import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# Loading your CSV data into a pandas DataFrame
df = pd.read_csv('train_data.csv')  # Replace 'your_data.csv' with your actual file name

##### Dealing with non-numerical values in the Secondary structure column
# List of possible values in the 'Secondary_structure' column
Second_strct_values = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
# Create a dictionary with the values as keys and their index as the value
Second_strct_dict = {value: index for index, value in enumerate(Second_strct_values)}

if 'Secondary_Structure' in df.columns:
    df['Secondary_Structure'] = df['Secondary_Structure'].map(Second_strct_dict)
    df['Secondary_Structure'] = df['Secondary_Structure'].fillna(7)   # Replace NaN with 7    

#### Dealing with non-numerical values from Neighbor column
def extract_single_neighbor_feature(df):
    """
    Extract a single numerical feature from the 'Neighbors' column 
    that captures the most important information about protein binding sites.
    
    This function creates a weighted neighbor score that considers:
    1. Number of neighbors
    2. Chain diversity (with higher weight for different chain interactions)
    3. Proximity of neighbors (closer neighbors have higher influence)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'Neighbors' column with strings like "C:68,A:2,A:4,A:5,A:3"
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with a new 'Neighbor_Score' column and 'Neighbors' column removed
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    def calculate_neighbor_score(row):
        neighbor_str = row['Neighbors']
        # Return 0 if Neighbors is null or empty
        if not pd.notnull(neighbor_str) or not str(neighbor_str).strip():
            return 0
            
        # Parse the neighbor string
        neighbors = str(neighbor_str).split(',')
        
        # Early return if no neighbors
        if not neighbors or neighbors[0] == '':
            return 0
            
        # Initialize counters
        num_neighbors = len(neighbors)
        chains = set()
        same_chain_count = 0
        diff_chain_count = 0
        
        # Get the current chain if available
        current_chain = row.get('Chain', None)
        
        # Process each neighbor
        for item in neighbors:
            if ':' in item:
                parts = item.split(':')
                if len(parts) == 2:
                    chain = parts[0]
                    chains.add(chain)
                    
                    # Count same vs different chain interactions
                    if current_chain is not None:
                        if chain == current_chain:
                            same_chain_count += 1
                        else:
                            diff_chain_count += 1
        
        # Calculate chain diversity factor (higher weight for different chain interactions)
        chain_diversity = len(chains) / max(1, num_neighbors)
        
        # Calculate weighted score
        # Different chain interactions are weighted 3 times more than same chain
        weighted_interactions = (same_chain_count + 3 * diff_chain_count) / max(1, num_neighbors)
        
        # Final score combines number of neighbors and their diversity
        neighbor_score = num_neighbors * (0.3 + 0.7 * (chain_diversity * weighted_interactions))
        
        return neighbor_score
    
    # Calculate the neighbor score for each row
    result_df['Neighbor_Score'] = result_df.apply(calculate_neighbor_score, axis=1)
    
    # Drop the original Neighbors column - THIS IS IMPORTANT
    result_df = result_df.drop(columns=['Neighbors'])
    
    return result_df
result_df= extract_single_neighbor_feature(df)

# Save the modified DataFrame
result_df.to_csv('train_data_transformed.csv', index=False)
