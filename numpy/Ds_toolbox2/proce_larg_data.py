# Initialize an empty dictionary counts_dict for storing the results of processing the Twitter data.
# Iterate over the 'tweets.csv' file by using a for loop. Use the loop variable chunk and iterate over the call to pd.read_csv() with a chunksize of 10.
# In the inner loop, iterate over the column 'lang' in chunk by using a for loop. Use the loop variable entry.



# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)
