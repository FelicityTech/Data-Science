# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)



# Create a zip object by using zip() on mutants and powers, in that order. Assign the result to z1.
# Print the tuples in z1 by unpacking them into positional arguments using the * operator in a print() call.
# Because the previous print() call would have exhausted the elements in z1, recreate the zip object you defined earlier and assign the result again to z1.
# 'Unzip' the tuples in z1 by unpacking them into positional arguments using the * operator in a zip() call. Assign the results to result1 and result2, in that order.
# The last print() statements prints the output of comparing result1 to mutants and result2 to powers. Click Submit Answer to see if the unpacked result1 and result2 are equivalent to mutants and powers, respectively.

