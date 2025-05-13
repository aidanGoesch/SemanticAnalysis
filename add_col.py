import csv
import sys
import pathlib

def main(model_name: str):
    """Adds memType column from hcV3-stories.csv to main.csv files"""
    # Load memType data
    memtype_data = {}
    with open("./datasets/hcV3-stories.csv", 'r') as f:
        for row in csv.DictReader(f):
            memtype_data[row['AssignmentId']] = row['memType']
    
    # Update all target files
    for i in range(1, 10):
        csv_path = pathlib.Path(f"./outputs/{model_name}/{i}/main.csv")
        if not csv_path.exists():
            continue
            
        # Read data
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames + (['memType'] if 'memType' not in reader.fieldnames else [])
            rows = list(reader)
        
        # Add memType column
        for row in rows:
            if row['AssignmentId'] in memtype_data:
                row['memType'] = memtype_data[row['AssignmentId']]
        
        # Write back
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    """
    example usage
    python3 add_col.py gpt-2-xl
    """
    model_name = sys.argv[1]
    main(model_name)
    print("Column Added!")