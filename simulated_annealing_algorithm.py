import math
import random

from matplotlib import pyplot as plt


def objective_function(target, current):
    """
    Calculate the sum of absolute differences between the target and current images.

    Parameters:
    target (list of list of int): The target binary image matrix.
    current (list of list of int): The current binary image matrix.

    Returns:
    int: The sum of absolute differences between corresponding pixels of the target and current images.
    """
    assert (len(target) == len(current)
            and all(len(row_t) == len(row_c)
                    for row_t, row_c in
                    zip(target, current))), "Target and current images must have the same dimensions"
    return sum(abs(t - c) for row_t, row_c in zip(target, current) for t, c in zip(row_t, row_c))


def random_matrix(width, height):
    """
    Generate a random binary matrix of specified dimensions.

    Parameters:
    width (int): Width of the matrix.
    height (int): Height of the matrix.

    Returns:
    list of list of int: A randomly generated binary matrix with given dimensions.
    """
    assert width > 0 and height > 0, "Width and height must be positive integers"
    return [[random.randint(0, 1) for _ in range(width)] for _ in range(height)]


def flip_pixel(image, x, y):
    """
    Flip the value of a pixel in the image at the specified coordinates.

    Parameters:
    image (list of list of int): The binary image matrix.
    x (int): X-coordinate of the pixel to flip.
    y (int): Y-coordinate of the pixel to flip.

    Returns:
    list of list of int: A new image matrix with the specified pixel flipped.
    """
    assert 0 <= x < len(image[0]) and 0 <= y < len(image), "x and y must be within the bounds of the image"
    new_image = [row[:] for row in image]
    new_image[y][x] = 1 - new_image[y][x]
    return new_image


def print_image(image):
    """
    Print the binary image to the console.

    Parameters:
    image (list of list of int): The binary image matrix to be printed.
    """
    for row in image:
        for pixel in row:
            print('1' if pixel == 1 else ' ', end='')
        print()
    print()


def visualazing_image(scores):
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
    plt.plot(scores)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Value Over Iterations')
    plt.show()


def simulated_annealing(target,
                        width,
                        height,
                        max_iterations=1000,
                        initial_temp=10,
                        ):
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
    assert all(len(row) == width for row in
               target), "All rows in the target image must have the same width as the specified width"
    assert len(target) == height, "The height of the target image must match the specified height"
    assert max_iterations > 0, "Max iterations must be a positive integer"
    assert initial_temp > 0, "Initial temperature must be positive"

    current_image = random_matrix(width, height)
    current_score = objective_function(target, current_image)
    best_image = current_image
    best_score = current_score
    scores = []
    temp = initial_temp

    for iteration in range(max_iterations):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        candidate_image = flip_pixel(current_image, x, y)
        candidate_score = objective_function(target, candidate_image)

        if candidate_score < best_score:
            best_image, best_score = candidate_image, candidate_score
            scores.append(current_score)

        difference = candidate_score - current_score
        t = temp / float(iteration + 1)
        metropolis = math.exp(-difference / t)

        if difference < 0 or random.random() < metropolis:
            current_image, current_score = candidate_image, candidate_score

        print(f'Iteration number {iteration}: Best eval = {best_score}')
        print_image(best_image)

    return best_image, best_score, scores


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
visualazing_image(scores)
