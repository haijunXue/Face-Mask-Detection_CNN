import numpy as np

# y = wx + b
def compute_error_for_line_given_points(w, b, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / len(points)

def step_gradient(b_current, w_current, points, learningRate):
    w_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Calculate gradients
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
    # Update w and b
    new_w = w_current - (learningRate * w_gradient)
    new_b = b_current - (learningRate * b_gradient)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=',', skip_header=1)
    learning_rate = 0.0001
    initial_w = 0
    initial_b = 0
    num_iterations = 1000
    print("Starting gradient descent at initial_b = {0}, initial_w = {1}, the initial error = {2}".format(
        initial_b, initial_w, compute_error_for_line_given_points(initial_w, initial_b, points)
    ))

    print("Running...")
    b, w = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations, b = {1}, w = {2}, the final error = {3}".format(
        num_iterations, b, w, compute_error_for_line_given_points(w, b, points)
    ))

if __name__ == '__main__':
    run()
