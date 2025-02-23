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


if __name__ == "__main__":
    top_stories = analyze_embeddings()
    save_top_stories(top_stories)




