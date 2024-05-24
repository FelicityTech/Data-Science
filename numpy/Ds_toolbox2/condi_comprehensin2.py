# You will work on the same list, fellowship and, using a list comprehension and an if-else conditional statement in the output expression, create a list that keeps members of fellowship with 7 or more characters and replaces others with an empty string. Use member as the iterator variable in the list comprehension.
# In the output expression, keep the string as-is if the number of characters is >= 7, else replace it with an empty string - that is, '' or "".

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [ member if len(member)>= 7 else '' for member in fellowship]

# Print the new list
print(new_fellowship)