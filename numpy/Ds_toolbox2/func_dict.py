# Define the function lists2dict() with two parameters: first is list1 and second is list2.
# Return the resulting dictionary rs_dict in lists2dict().
# Call the lists2dict() function with the arguments feature_names and row_vals. Assign the result of the function call to rs_fxn.



# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)

# Print rs_fxn
print(rs_fxn)