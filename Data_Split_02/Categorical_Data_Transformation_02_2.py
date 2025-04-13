import pandas as pd

df = pd.read_csv('train_data.csv')  

##### Dealing with non-numerical values in the Secondary structure column
Second_strct_values = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
Second_strct_dict = {value: index for index, value in enumerate(Second_strct_values)}

if 'Secondary_Structure' in df.columns:
    df['Secondary_Structure'] = df['Secondary_Structure'].map(Second_strct_dict)
    df['Secondary_Structure'] = df['Secondary_Structure'].fillna(7)      

#### Dealing with non-numerical values from Neighbor column
def extract_single_neighbor_feature(df):
    """Calculate and extract a single numerical score from the neighbors feature;
    this calculation is based on the number of neighbors of each residue, the chain
    and the proximity of other neighbors; the original column for neighbors is droped
    in order to avoid inconsistences.""" 

    result_df = df.copy()
    
    def calculate_neighbor_score(row):
        neighbor_str = row['Neighbors']
        if not pd.notnull(neighbor_str) or not str(neighbor_str).strip():
            return 0
            
        neighbors = str(neighbor_str).split(',')
        
        if not neighbors or neighbors[0] == '':
            return 0
            
        num_neighbors = len(neighbors)
        chains = set()
        same_chain_count = 0
        diff_chain_count = 0
        
        current_chain = row.get('Chain', None)
        
        for item in neighbors:
            if ':' in item:
                parts = item.split(':')
                if len(parts) == 2:
                    chain = parts[0]
                    chains.add(chain)
                    
                    if current_chain is not None:
                        if chain == current_chain:
                            same_chain_count += 1
                        else:
                            diff_chain_count += 1
        
        chain_diversity = len(chains) / max(1, num_neighbors)
        
        weighted_interactions = (same_chain_count + 3 * diff_chain_count) / max(1, num_neighbors)
        
        neighbor_score = num_neighbors * (0.3 + 0.7 * (chain_diversity * weighted_interactions))
        
        return neighbor_score
    
    result_df['Neighbor_Score'] = result_df.apply(calculate_neighbor_score, axis=1)
    
    # Drop the original neighbors column 
    result_df = result_df.drop(columns=['Neighbors'])
    
    return result_df

result_df= extract_single_neighbor_feature(df)

result_df.to_csv('train_data_transformed.csv', index=False)
