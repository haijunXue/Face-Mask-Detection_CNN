import numpy as np


# y = wx + b
def compute_error_for_line_given_points(w, b, points):
    totalError = np.sum((points[:, 1] - (w * points[:, 0] + b)) ** 2)
    return totalError / len(points)


def step_gradient(b_current, w_current, points, learningRate):
    N = float(len(points))
    x = points[:, 0]
    y = points[:, 1]

    # Vectorized operations
    prediction = w_current * x + b_current
    error = y - prediction

    # Calculate gradients
    w_gradient = -(2 / N) * np.sum(x * error)
    b_gradient = -(2 / N) * np.sum(error)

    # Gradient clipping to avoid overflow
    w_gradient = np.clip(w_gradient, -1e5, 1e5)
    b_gradient = np.clip(b_gradient, -1e5, 1e5)

    # Update weights
    new_w = w_current - (learningRate * w_gradient)
    new_b = b_current - (learningRate * b_gradient)

    return new_b, new_w


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learning_rate)
        # Optionally, print intermediate results for debugging
        if i % 100 == 0:
            print(f"Iteration {i}: b = {b}, w = {w}, error = {compute_error_for_line_given_points(w, b, points)}")
    return b, w


def run():
    # Load and normalize data
    points = np.genfromtxt("data.csv", delimiter=',', skip_header=1)
    points[:, 0] = (points[:, 0] - np.mean(points[:, 0])) / np.std(points[:, 0])  # Normalize x

    learning_rate = 0.00001  # Lower learning rate
    initial_w = 0
    initial_b = 0
    num_iterations = 1000

    print("Starting gradient descent with initial_b = {0}, initial_w = {1}, initial error = {2}".format(
        initial_b, initial_w, compute_error_for_line_given_points(initial_w, initial_b, points)
    ))

    print("Running...")
    b, w = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations, b = {1}, w = {2}, final error = {3}".format(
        num_iterations, b, w, compute_error_for_line_given_points(w, b, points)
    ))


if __name__ == '__main__':
    run()
