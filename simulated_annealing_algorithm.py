import math
import random
from typing import List, Tuple

from matplotlib import pyplot as plt


# Defines the objective function for the simulated annealing algorithm
def objective_function(target: List[int], current: List[int]) -> int:
    """
    Calculate the sum of absolute differences between the target and current images.

    Parameters:
    target (list of list of int): The target binary image matrix.
    current (list of list of int): The current binary image matrix.

    Returns:
    int: The sum of absolute differences between corresponding pixels of the target and current images.
    """
    # Asserts that the target and current images have the same dimensions
    assert (len(target) == len(current)
            and all(len(row_t) == len(row_c)
                    for row_t, row_c in
                    zip(target, current))), "Target and current images must have the same dimensions"
    # Calculates and returns the sum of absolute differences
    return sum(abs(t - c) for row_t, row_c in zip(target, current) for t, c in zip(row_t, row_c))


# Generates a random binary matrix of given dimensions
def random_matrix(width: int, height: int) -> List[List[int]]:
    """
    Generate a random binary matrix of specified dimensions.

    Parameters:
    width (int): Width of the matrix.
    height (int): Height of the matrix.

    Returns:
    list of list of int: A randomly generated binary matrix with given dimensions.
    """
    # Ensures that the width and height are positive integers
    assert width > 0 and height > 0, "Width and height must be positive integers"
    # Return a matrix with random 0s and 1s
    return [[random.randint(0, 1) for _ in range(width)] for _ in range(height)]


# Flips the value of a specified pixel in an image
def flip_pixel(image: List[List[int]], x: int, y: int) -> List[List[int]]:
    """
    Flip the value of a pixel in the image at the specified coordinates.

    Parameters:
    image (list of list of int): The binary image matrix.
    x (int): X-coordinate of the pixel to flip.
    y (int): Y-coordinate of the pixel to flip.

    Returns:
    list of list of int: A new image matrix with the specified pixel flipped.
    """
    # Ensure the pixel coordinates are within the image dimensions
    assert 0 <= x < len(image[0]) and 0 <= y < len(image), "x and y must be within the bounds of the image"
    new_image = [row[:] for row in image]  # Create a copy of the image
    new_image[y][x] = 1 - new_image[y][x]  # Flip the pixel value
    return new_image  # Return the modified image


# Prints the binary image to the console
def print_image(image: List[List[int]]):
    """
    Print the binary image to the console.

    Parameters:
    image (list of list of int): The binary image matrix to be printed.
    """
    for row in image:  # Iterate over each row in the image
        for pixel in row:  # Iterate over each pixel in the row
            print('1' if pixel == 1 else ' ', end='')  # Print '1' for pixel value 1, otherwise print a space
        print()  # Print a newline at the end of each row
    print()  # Print an extra newline for separation


# Plots the evolution of the objective function values over iterations
def visualizing_objective_func(scores: List[int]):
    """
    Plot the evolution of the objective function values over iterations.

    This function takes a list of scores (objective function values) and plots them against the iteration number.
    It is used to visualize the performance of the simulated annealing algorithm over time, showing how the
    objective function value changes as the algorithm progresses.

    Parameters:
    scores (list of int): A list of integer scores representing the objective function values at each iteration.

    Returns:
    None: This function does not return anything but displays a matplotlib plot.
    """
    plt.plot(scores)   # Plot the scores
    plt.xlabel('Iteration')  # Label the x-axis as 'Iteration'
    plt.ylabel('Objective Function Value')  # Label the y-axis as 'Objective Function Value'
    plt.title('Objective Function Value Over Iterations')  # Set the title of the plot
    plt.show()  # Display the plot


# Define the simulated annealing function# Performs the simulated annealing algorithm
def simulated_annealing(target: List[List[int]],
                        width: int,
                        height: int,
                        max_iterations: int = 1000,
                        initial_temp: int = 20,
                        ) -> Tuple[List[List[int]], int, List[int]]:
    """
    Perform the simulated annealing algorithm to find an image matrix similar to the target.

    Parameters:
    target (list of list of int): The target binary image matrix.
    width (int): Width of the image matrix.
    height (int): Height of the image matrix.
    max_iterations (int): The maximum number of iterations to run the algorithm.
    initial_temp (float): The initial temperature for the annealing process.

    Returns:
    tuple:
        - list of list of int: The best image matrix found.
        - int: The score of the best image.
        - list of int: The scores of the best image over iterations.
    """
    # Validate the input dimensions and parameters
    assert all(len(row) == width for row in
               target), "All rows in the target image must have the same width as the specified width"
    assert len(target) == height, "The height of the target image must match the specified height"
    assert max_iterations > 0, "Max iterations must be a positive integer"
    assert initial_temp > 0, "Initial temperature must be positive"

    current_image = random_matrix(width, height)  # Generate a random initial image
    current_score = objective_function(target, current_image)  # Calculate the initial score
    best_image = current_image  # Initialize the best image as the current image
    best_score = current_score  # Initialize the best score as the current score
    scores = []  # List to keep track of scores at each iteration
    temp = initial_temp  # Set the initial temperature

    # Iterate over the maximum number of iterations
    for iteration in range(max_iterations):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)  # Choose a random pixel to flip
        candidate_image = flip_pixel(current_image, x, y)  # Flip the chosen pixel
        candidate_score = objective_function(target, candidate_image)  # Calculate the score of the candidate image

        # Update the best image and score if the candidate is better
        if candidate_score < best_score:
            best_image, best_score = candidate_image, candidate_score
            scores.append(current_score)  # Append the current score to the scores list

        # Calculate the difference in score and the temperature for this iteration
        difference = candidate_score - current_score
        t = temp / float(iteration + 1)
        # Calculate the probability of accepting the candidate image
        metropolis = math.exp(-difference / t)

        # Decide whether to accept the candidate image
        if difference < 0 or random.random() < metropolis:
            current_image, current_score = candidate_image, candidate_score

        # Print the current iteration number and the best score
        print(f'Iteration number {iteration}: Best eval = {best_score}')
        # Print the current best image
        print_image(best_image)

        # Check if the current score is 0 or less, and break the loop if so
        if current_score <= 0.0:
            print(f'Score reached 0 at iteration {iteration}, exiting the loop')
            break

    # Return the best image found, its score, and the list of scores
    return best_image, best_score, scores


if __name__ == "__main__":
    # Set a random seed for reproducibility
    random.seed(1)

    # Define a target image as a 10x10 binary matrix
    target_image = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    # Extract width and height from the target image
    width, height = len(target_image[0]), len(target_image)

    # Execute the simulated annealing algorithm
    final_image, final_score, scores = simulated_annealing(target_image, width, height)

    # Display the final results
    print('Final image:')
    print_image(final_image)
    print('Final score:', final_score)

    # Plot the evolution of the objective function over iterations
    visualizing_objective_func(scores)
