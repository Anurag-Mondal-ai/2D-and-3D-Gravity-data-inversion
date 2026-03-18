import numpy as np

def rand_spl(x_u, i):
    """
    Generate 3 unique random integers from 1 to x_u (inclusive), excluding integer i.

    Parameters:
    x_u : int
        Upper limit for random numbers (inclusive).
    i : int
        Integer to exclude from the random selection.

    Returns:
    sp_rand : ndarray
        Array of 3 random integers excluding i.
    """
    if x_u <= 1:
        raise ValueError("x_u must be greater than 1 to allow exclusions.")
    if x_u - 1 < 3:
        raise ValueError("Not enough elements to choose from after excluding 'i'.")
    
    # All integers from 1 to x_u, excluding i
    candidates = [x for x in range(1, x_u + 1) if x != i]

    # Randomly choose 3 unique values
    sp_rand = np.random.choice(candidates, size=3, replace=False)

    return sp_rand
print(rand_spl(10, 2))  # Output: 3 random integers from 1 to 10, excluding 2
