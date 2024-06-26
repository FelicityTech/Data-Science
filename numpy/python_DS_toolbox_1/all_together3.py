# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.')

    # Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1
        
        # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Print result1
print(result1)



# If col_name is not a column in the DataFrame df, raise a ValueError 'The DataFrame does not have a ' + col_name + ' column.'.
# Call your new function count_entries() to analyze the 'lang' column of tweets_df. Store the result in result1.
# Print result1. This has been done for you, so hit 'Submit Answer' to check out the result. In the next exercise, you'll see that it raises the necessary ValueErrors.