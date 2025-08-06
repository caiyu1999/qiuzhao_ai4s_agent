# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place circles around it in a hexagonal pattern using a more optimal radius distribution
    for i in range(12):
        angle = 2 * np.pi * i / 12
        radius = 0.2 + 0.1 * (i % 2)  # Alternating radius for better packing
        centers[i + 1] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]

    # Place 12 additional circles in a second layer with varied radii
    for i in range(12):
        angle = 2 * np.pi * i / 12
        radius = 0.4 + 0.1 * (i % 2)  # Alternate radii for denser packing
        centers[i + 13] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]

    # Ensure all circles are inside the square with a safer margin
    centers = np.clip(centers, 0.1, 0.9)  # Adjusted bounds for circles' radii

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)  # Initialize to zero to avoid radius overlap issues

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y) * 0.95  # Adding a margin to avoid boundary issues

    # Then, limit by distance to other circles more efficiently
    # Using a set for faster overlap checking
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])

            # If current radii would cause overlap, adjust only if necessary
            if radii[i] + radii[j] > dist:
                # Scale radii to avoid overlap
                min_scale = dist / (radii[i] + radii[j])
                radii[i] *= min_scale
                radii[j] *= min_scale

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
