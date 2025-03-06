import tensorflow_hub as hub
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler

# load data
def load_data():
    df = pd.read_csv("./data/hcV3-stories.csv")
    return df

def divide_data(df:pd.DataFrame) -> np.array:
    df['bin'] = df.apply(determine_bin, axis=1)
    
    # Create dictionary to store the divided data
    divided_data = {
        'Recall': df[df['bin'] == 'Recall'],
        'Retold': df[df['bin'] == 'Retold'],
        'Imagined': df[df['bin'] == 'Imagined'],
        'Error': df[df['bin'] == 'Error']
    }
    
    return divided_data

def determine_bin(row):
    """Function used to divide each value into bins according to whether it was an imagined, or remembered"""
    # recAgnPairId = None for imagined stories -> ID of the retold story being recalled
    # if not None then its recalled

    # recImgPairId = None for retold stories -> ID of imagined story being recalled
    # if not None then its recalled

    if isinstance(row['recAgnPairId'], float) and isinstance(row['recImgPairId'], float):
        return 'Recall'  # This is a recall (neither points to it)
    elif not isinstance(row['recAgnPairId'], float) and isinstance(row['recImgPairId'], float):
        return 'Retold'  # This is a retold story with a recall
    elif isinstance(row['recAgnPairId'], float) and not isinstance(row['recImgPairId'], float):
        return 'Imagined'  # This is an imagined story with a recall
    else:
        return 'Error'  # Can't be both retold and imagined

def analyze_embeddings():
    # Load data
    df = load_data()
    divided_data = divide_data(df)
    
    print("Loading embedding model...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Embedding model loaded successfully")
    
    top_stories = {}  # Dictionary to store top stories for each bin
    
    print("Total number of rows:", len(df))
    for bin_name, subset in divided_data.items():
        print(f"\nProcessing {bin_name} subset (size: {len(subset)})")
        
        # Get stories and their AssignmentIds
        stories = subset['story'].tolist()
        assignment_ids = subset['AssignmentId'].tolist()
        
        # Calculate embeddings for all stories in the subset
        embeddings = embed(stories).numpy()
        
        # Calculate average vector
        avg_vec = np.mean(embeddings, axis=0)
        
        # Calculate similarities and store with AssignmentIds
        similarities_with_ids = []
        for i, vec in enumerate(embeddings):
            similarity = np.dot(vec, avg_vec) / (norm(vec) * norm(avg_vec))
            similarities_with_ids.append({
                'AssignmentId': assignment_ids[i],
                'similarity': similarity
            })
        
        # Sort by similarity and get top 100
        sorted_stories = sorted(similarities_with_ids, 
                              key=lambda x: x['similarity'], 
                              reverse=True)[:100]
        
        # Get the original rows for the top 100 stories
        top_assignment_ids = [story['AssignmentId'] for story in sorted_stories]
        top_stories[bin_name] = subset[subset['AssignmentId'].isin(top_assignment_ids)].copy()
        
        # Add similarity scores to the DataFrame
        similarity_dict = {story['AssignmentId']: story['similarity'] 
                         for story in sorted_stories}
        top_stories[bin_name]['similarity_score'] = \
            top_stories[bin_name]['AssignmentId'].map(similarity_dict)
        
        # Sort by similarity score
        top_stories[bin_name] = top_stories[bin_name].sort_values(
            'similarity_score', ascending=False)
        
        print(f"Top 5 similarity scores for {bin_name}:")
        print(top_stories[bin_name]['similarity_score'].head())

    return top_stories

def save_top_stories(top_stories):
    """Save top stories from each bin to separate CSV files"""
    for bin_name, stories_df in top_stories.items():
        filename = f"top_100_{bin_name.lower()}_stories.csv"
        stories_df.to_csv(filename, index=False)
        print(f"Saved {bin_name} stories to {filename}")


def merge_top_stories():
    """
    Merges the top stories CSV files into a single CSV with an additional column indicating the bin.
    Returns the merged DataFrame and saves it to a file.
    """
    # List of bin names we expect
    bin_names = ['Recall', 'Retold', 'Imagined']
    
    # Initialize list to store DataFrames
    dfs = []
    
    # Read each CSV and add bin identifier
    for bin_name in bin_names:
        try:
            df = pd.read_csv(f"top_100_{bin_name.lower()}_stories.csv")
            df['story_type'] = bin_name  # Add column to identify which bin it came from
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: Could not find file for {bin_name}")
            continue
    
    # Merge all DataFrames
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by similarity score to see best matches across all categories
        merged_df = merged_df.sort_values('similarity_score', ascending=False)
        
        # Save merged data
        output_file = "merged_top_stories.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Merged data saved to {output_file}")
        
        # Print summary statistics
        print("\nSummary of merged data:")
        print(f"Total rows: {len(merged_df)}")
        print("\nStories per type:")
        print(merged_df['story_type'].value_counts())
        print("\nAverage similarity score by type:")
        print(merged_df.groupby('story_type')['similarity_score'].mean())
        
        return merged_df
    else:
        print("No files were found to merge.")
        return None


def make_large_subset(df:pd.DataFrame):
    """
    Function that returns a random subset of data that makes up a quarter of the original DataFrame
    """
    # 1/4 of the data
    sample_size = df.shape[0] // 4

    subset = df.sample(n=sample_size, replace=False)

    subset.to_csv("./datasets/hcV3-stories-quartered.csv")

    return subset


def make_proportional_subset_using_other_subset(data, other):
    """ 
    Function that will make a proportional subset of `data` that incorporates data from `other`.
    
    Args:
        data (str): Path to the main CSV file
        other (str): Path to the CSV file containing data that must be incorporated
        
    Returns:
        pandas.DataFrame: A new subset of the data that:
            1. Has no duplicates with 'other'
            2. When combined with 'other', is representative of original data
            3. Has size of 1000 - len(other)
    """
    import pandas as pd
    import numpy as np
    
    # Load both datasets
    main_data = pd.read_csv(data)
    other_data = pd.read_csv(other)
    
    # Apply bin categorization
    main_data['bin'] = main_data.apply(determine_bin, axis=1)
    other_data['bin'] = other_data.apply(determine_bin, axis=1)
    
    # Get original proportions
    original_proportions = main_data['bin'].value_counts(normalize=True).to_dict()
    
    # Calculate sizes
    other_size = len(other_data)
    target_size = 1000
    new_subset_size = target_size - other_size
    
    if new_subset_size <= 0:
        raise ValueError(f"The 'other' dataset already has {other_size} rows, which is >= 1000")
    
    # Remove rows from main_data that are already in other_data (based on AssignmentId)
    filtered_main_data = main_data[~main_data['AssignmentId'].isin(other_data['AssignmentId'])]
    
    # Calculate how many from each bin should be in the combined dataset
    total_target_counts = {bin_type: int(round(prop * target_size)) 
                          for bin_type, prop in original_proportions.items()}
    
    # Get the current counts in the other dataset
    other_counts = other_data['bin'].value_counts().to_dict()
    
    # Calculate how many we need to add for each bin
    needed_counts = {}
    for bin_type in original_proportions.keys():
        other_count = other_counts.get(bin_type, 0)
        # If we already have more than the proportional amount in 'other', don't add more
        if other_count >= total_target_counts[bin_type]:
            needed_counts[bin_type] = 0
        else:
            needed_counts[bin_type] = total_target_counts[bin_type] - other_count
    
    # Calculate current total and adjust if needed
    total_needed = sum(needed_counts.values())
    
    if total_needed != new_subset_size:
        # Calculate a scaling factor to adjust all bins proportionally
        scaling_factor = new_subset_size / total_needed if total_needed > 0 else 0
        
        # Apply scaling and round to integers
        for bin_type in needed_counts:
            needed_counts[bin_type] = int(round(needed_counts[bin_type] * scaling_factor))
        
        # Handle any remaining difference with largest or smallest bin
        adjusted_total = sum(needed_counts.values())
        diff = new_subset_size - adjusted_total
        
        if diff != 0:
            # Sort bins by count, descending if adding, ascending if removing
            sorted_bins = sorted(needed_counts.keys(), 
                                key=lambda x: needed_counts[x],
                                reverse=(diff > 0))
            
            for bin_type in sorted_bins:
                if needed_counts[bin_type] > 0 or diff > 0:
                    adjust = 1 if diff > 0 else -1
                    needed_counts[bin_type] += adjust
                    diff -= adjust
                    if diff == 0:
                        break
    
    # Create the new subset by sampling from each bin
    new_subset = pd.DataFrame()
    for bin_type, count in needed_counts.items():
        if count <= 0:
            continue
        
        bin_data = filtered_main_data[filtered_main_data['bin'] == bin_type]
        
        if len(bin_data) <= count:
            selected = bin_data
            print(f"Warning: Not enough data for bin {bin_type}. Requested {count}, got {len(bin_data)}")
        else:
            selected = bin_data.sample(count)
        
        new_subset = pd.concat([new_subset, selected])
    
    # Final check to ensure exact size
    if len(new_subset) != new_subset_size:
        diff = new_subset_size - len(new_subset)
        if diff > 0:
            remaining_data = filtered_main_data[~filtered_main_data['AssignmentId'].isin(new_subset['AssignmentId'])]
            if len(remaining_data) >= diff:
                additional = remaining_data.sample(diff)
                new_subset = pd.concat([new_subset, additional])
        elif diff < 0:
            new_subset = new_subset.sample(new_subset_size)
    
    # Verify the results
    assert len(new_subset) == new_subset_size, f"New subset has {len(new_subset)} rows, expected {new_subset_size}"
    assert len(set(new_subset['AssignmentId']).intersection(set(other_data['AssignmentId']))) == 0, "Duplicate AssignmentIds found"
    
    # Debug: Check proportions
    combined = pd.concat([other_data, new_subset])
    combined_props = combined['bin'].value_counts(normalize=True).to_dict()
    
    print("Original proportions:")
    for bin_type in original_proportions:
        print(f"  {bin_type}: {original_proportions[bin_type]:.4f}")
    
    print("\nCombined proportions:")
    for bin_type in original_proportions:
        combined_prop = combined_props.get(bin_type, 0)
        print(f"  {bin_type}: {combined_prop:.4f}")
        print(f"  Difference: {combined_prop - original_proportions[bin_type]:.4f}")
    
    # Drop the 'bin' column before returning
    if 'bin' in new_subset.columns:
        new_subset = new_subset.drop('bin', axis=1)
    
    return new_subset


if __name__ == "__main__":
    top_stories = analyze_embeddings()
    save_top_stories(top_stories)




