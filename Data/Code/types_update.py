import pandas as pd

def transfer_column(source_csv, target_csv, output_csv, id_column, column_to_transfer):
    """
    Transfers a column from one CSV to another based on a matching ID column.

    Args:
        source_csv (str): Path to the source CSV file.
        target_csv (str): Path to the target CSV file.
        output_csv (str): Path for the output CSV file.
        id_column (str): The name of the ID column used for matching.
        column_to_transfer (str): The name of the column to transfer from source to target.
    """
    # Load the source and target CSVs into DataFrames
    source_df = pd.read_csv(source_csv)
    target_df = pd.read_csv(target_csv)

    # Ensure the column to transfer exists in the source CSV
    if column_to_transfer not in source_df.columns:
        raise ValueError(f"Column '{column_to_transfer}' not found in source CSV.")

    # Merge the target DataFrame with the relevant column from the source DataFrame
    merged_df = target_df.merge(source_df[[id_column, column_to_transfer]], on=id_column, how='left')

    # Save the merged DataFrame to the output CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"Column '{column_to_transfer}' has been successfully transferred to {output_csv}.")

# Example usage
transfer_column(
    source_csv='backup.csv',
    target_csv='update.csv',
    output_csv='ouput.csv',
    id_column='id',
    column_to_transfer='location'
)
