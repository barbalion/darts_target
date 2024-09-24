import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
import multiprocessing as mp
import time
import sys
from matplotlib.colors import LinearSegmentedColormap

# Official dartboard dimensions in millimeters
# Total dartboard radius
outer_radius = 451 / 2  # mm

# Bullseye dimensions
inner_bull_radius = 12.7 / 2  # mm
outer_bull_radius = 31.8 / 2  # mm

# Triple ring dimensions
triple_ring_outer_radius = 214 / 2  # mm
triple_ring_width = 8  # mm
triple_ring_inner_radius = triple_ring_outer_radius - triple_ring_width  # mm

# Double ring dimensions
double_ring_outer_radius = 340 / 2  # mm
double_ring_width = 8  # mm
double_ring_inner_radius = double_ring_outer_radius - double_ring_width  # mm

lines_color = 'darkgray'

# Colors for the rings
ring_colors = ['red', 'green']  # For double and triple rings

# Colors for the main scoring areas
main_colors = ['black', 'white']  # For single score areas

# Numbers on the dartboard
numbers = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

num_segments = len(numbers)
angles = np.linspace(
    1 / 2 + 1 / num_segments, -3 / 2 + 1 / num_segments,
    num_segments + 1, endpoint=True
) * np.pi


# Function to normalize angle
def norm_angle(a, min_a=0.0):
    while a < min_a:
        a += 2 * np.pi
    while a >= min_a + 2 * np.pi:
        a -= 2 * np.pi
    return a


# Function to calculate the score based on (x, y) coordinates
def calculate_score(x, y):
    # Calculate the distance from the center
    r = np.sqrt(x ** 2 + y ** 2)

    # Check if the point is outside the dartboard
    if r > double_ring_outer_radius:
        return 0  # No score

    # Calculate the angle in radians and normalize
    theta = norm_angle(np.arctan2(y, x))

    # Find the corresponding segment number
    segment_index = None
    for i in range(num_segments):
        angle_start = norm_angle(angles[i + 1])
        angle_end = norm_angle(angles[i], angle_start)

        if angle_start <= norm_angle(theta, angle_start) <= angle_end:
            segment_index = i
            break

    if segment_index is None:
        segment_index = 0  # Default to first segment if not found

    base_score = numbers[segment_index]

    # Determine the multiplier based on the radius
    if r <= inner_bull_radius:
        return 50  # Inner bullseye
    elif r <= outer_bull_radius:
        return 25  # Outer bullseye
    elif triple_ring_inner_radius < r <= triple_ring_outer_radius:
        return base_score * 3  # Triple ring
    elif double_ring_inner_radius < r <= double_ring_outer_radius:
        return base_score * 2  # Double ring
    else:
        return base_score  # Single score area


# Function to calculate the average score for a grid point
def average_score(args):
    xi, yi, sigma, num_samples = args

    # Generate random points normally distributed around (xi, yi)
    x_samples = np.random.normal(xi, sigma, num_samples)
    y_samples = np.random.normal(yi, sigma, num_samples)

    # Calculate scores for all samples
    scores = np.array([
        calculate_score(x_sample, y_sample)
        for x_sample, y_sample in zip(x_samples, y_samples)
    ])

    # Return the average score
    return np.mean(scores)


# Function to calculate scores grid for a given sigma
def calculate_scores_grid(X, Y, sigma, num_samples):
    # Flatten the grid arrays for iteration
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Prepare the list of arguments
    args_list = [
        (X_flat[idx], Y_flat[idx], sigma, num_samples)
        for idx in range(len(X_flat))
    ]

    # Use multiprocessing for performance
    with mp.Pool() as pool:
        scores = pool.map(average_score, args_list)

    # Reshape the scores back to grid shape
    scores_grid = np.array(scores).reshape(X.shape)
    return scores_grid


# Function to draw the dartboard
def draw_dartboard(ax):
    # Draw the outer circle (edge of the dartboard)
    outer_circle = plt.Circle(
        (0, 0), outer_radius, color='none', ec='black', lw=2, zorder=2
    )
    ax.add_artist(outer_circle)

    # Draw the double ring (outermost ring)
    for i in range(num_segments):
        wedge = Wedge(
            (0, 0),
            double_ring_outer_radius,
            np.degrees(angles[i + 1]),
            np.degrees(angles[i]),
            width=double_ring_width,
            facecolor='none',
            ec=lines_color,
            lw=1,
            zorder=3
        )
        ax.add_artist(wedge)

    # Draw the main scoring areas
    for i in range(num_segments):
        # Main area between double ring inner and triple ring outer
        wedge_main = Wedge(
            (0, 0),
            double_ring_inner_radius,
            np.degrees(angles[i + 1]),
            np.degrees(angles[i]),
            width=double_ring_inner_radius - triple_ring_outer_radius,
            facecolor='none',
            ec=lines_color,
            lw=1,
            zorder=3
        )
        ax.add_artist(wedge_main)

        # Triple ring
        wedge_triple = Wedge(
            (0, 0),
            triple_ring_outer_radius,
            np.degrees(angles[i + 1]),
            np.degrees(angles[i]),
            width=triple_ring_width,
            facecolor='none',
            ec=lines_color,
            lw=1,
            zorder=3
        )
        ax.add_artist(wedge_triple)

        # Inner scoring area between triple ring inner and outer bull
        wedge_inner = Wedge(
            (0, 0),
            triple_ring_inner_radius,
            np.degrees(angles[i + 1]),
            np.degrees(angles[i]),
            width=triple_ring_inner_radius - outer_bull_radius,
            facecolor='none',
            ec=lines_color,
            lw=1,
            zorder=3
        )
        ax.add_artist(wedge_inner)

    # Draw the bullseye
    outer_bullseye = plt.Circle(
        (0, 0), outer_bull_radius, color='none', ec=lines_color, lw=1, zorder=4
    )
    inner_bullseye = plt.Circle(
        (0, 0), inner_bull_radius, color='none', ec=lines_color, lw=1, zorder=4
    )
    ax.add_artist(outer_bullseye)
    ax.add_artist(inner_bullseye)

    # Draw radial lines
    for a in angles:
        x_start = outer_bull_radius * np.cos(a)
        y_start = outer_bull_radius * np.sin(a)
        x_end = outer_radius * np.cos(a)
        y_end = outer_radius * np.sin(a)
        ax.plot(
            [x_start, x_end], [y_start, y_end],
            color=lines_color, lw=1, zorder=5
        )

    # Position the numbers outside the double ring
    number_ring_radius = double_ring_outer_radius + 20

    for i, a in enumerate((angles[:-1] + angles[1:]) / 2):
        x = number_ring_radius * np.cos(a)
        y = number_ring_radius * np.sin(a)
        ax.text(
            x, y, str(numbers[i]),
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='lightgray', zorder=6
        )

    # Adjust plot limits
    ax.set_xlim(-outer_radius - 100, outer_radius + 100)
    ax.set_ylim(-outer_radius - 100, outer_radius + 100)


if __name__ == "__main__":
    total_sigmas = 100
    start_time = time.time()

    # For estimating remaining time
    times = []

    for sigma in range(1, total_sigmas + 1, 4):
        iter_start_time = time.time()

        # Grid size adjusted based on sigma and outer_radius
        grid_size = int(min(max(outer_radius / sigma * 10, 80), outer_radius * 2))

        # Number of random points per grid point
        num_samples = int(sigma ** 2 * 10) + 1

        # Generate a grid over the dartboard area
        x = np.linspace(-outer_radius, outer_radius, grid_size)
        y = np.linspace(-outer_radius, outer_radius, grid_size)
        X, Y = np.meshgrid(x, y)

        # Calculate the scores grid
        scores_grid = calculate_scores_grid(X, Y, sigma, num_samples)

        # Mask the points outside the dartboard
        R = np.sqrt(X ** 2 + Y ** 2)
        mask = R > outer_radius
        scores_grid = np.ma.masked_where(mask, scores_grid)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.axis('off')  # Hide the axes

        # Overlay the heatmap over the dartboard
        extent = [-outer_radius, outer_radius, -outer_radius, outer_radius]

        # Calculate the 95th percentile value
        percentile_95 = np.percentile(scores_grid.compressed(), 95)

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [
            (0.00, (0, 0, 0)),  #  Black
            (0.06, (1, 0, 0)),  #  Red
            (0.12, (1, 1, 0)),  #  Yellow
            (0.16, (1, 1, 1)),  #  White
            (0.20, (0, 0, 1)),  #  Blue
            (0.30, (0, 1, 1)),  #  Cyan
            (1.00, (0, 1, 0)),  #  Green
        ])

        # Normalize the colormap
        norm = plt.Normalize(vmin=0, vmax=np.max(scores_grid))

        # Plot the heatmap first
        im = ax.imshow(
            scores_grid, extent=extent, origin='lower',
            cmap=cmap, alpha=0.8, zorder=1, vmin=0, vmax=60
        )

        # Draw the dartboard
        draw_dartboard(ax)

        # Add sigma value on the top-left corner
        ax.text(
            0.01, 0.99, f'sigma = {sigma} mm',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=14, fontweight='bold',
            color='white',
            bbox=dict(facecolor='black', edgecolor='none', pad=5, alpha=0.7),
            zorder=7
        )

        # Add colorbar to show the score values
        fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, label='Expected Score'
        )

        # Construct the filename with zero-padded sigma value
        filename = f'board{sigma:03d}.png'

        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        # Close the figure to free memory
        plt.close(fig)

        # Update progress
        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time - iter_start_time
        times.append(iter_elapsed_time)
        elapsed_time = iter_end_time - start_time
        avg_time_per_iter = sum(times) / len(times)
        remaining_iterations = total_sigmas - sigma
        est_remaining_time = avg_time_per_iter * remaining_iterations

        # Output progress
        print(
            f'Sigma {sigma}/{total_sigmas} completed. '
            f'Elapsed time: {elapsed_time:.2f} ({iter_elapsed_time:.2f} s for iteration). '
            f'Estimated time remaining: {est_remaining_time:.2f} s.'
        )
        sys.stdout.flush()
