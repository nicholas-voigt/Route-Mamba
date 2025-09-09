import torch


def greedy_initial_tour(batch: torch.Tensor):
    """
    Creates an initial tour for the TSP problem using a greedy nearest neighbor heuristic.
    Args:
        batch: (B, N, 2) - coordinates of nodes
    Returns: 
        (B, N, 2) - nodes in greedy order
    """
    B, N, _ = batch.size()
    MAX = torch.finfo(batch.dtype).max
    device = batch.device
    # Calculate full pairwise distance matrix (B, N, N)
    dist_matrix = torch.linalg.vector_norm(batch.unsqueeze(2) - batch.unsqueeze(1), dim=-1)
    # Initialize tour and masks
    route = torch.zeros(B, N, dtype=torch.long, device=device)
    visited_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    # Select start at node (0) for all instances in the batch
    current_node = torch.zeros(B, dtype=torch.long, device=device)
    route[:, 0] = current_node
    visited_mask[torch.arange(B), current_node] = False
    # Build tour iteratively
    for i in range(1, N):
        # Get distances from the current node to all other nodes & mask out visited
        current_dists = dist_matrix[torch.arange(B), current_node]
        current_dists[~visited_mask] = MAX
        # Find the nearest unvisited node & update route and mask
        current_node = torch.argmin(current_dists, dim=1)
        route[:, i] = current_node
        visited_mask[torch.arange(B), current_node] = False
    # Reorder original coordinates based on the computed route
    return batch.gather(1, route.unsqueeze(-1).expand(-1, -1, 2))

def farthest_initial_tour(batch: torch.Tensor):
    """
    Creates an initial tour for the TSP problem using a farthest neighbor heuristic.
    Args:
        batch: (B, N, 2) - coordinates of nodes
    Returns: 
        (B, N, 2) - nodes in farthest order
    """
    B, N, _ = batch.size()
    MIN = torch.finfo(batch.dtype).min
    device = batch.device
    # Calculate full pairwise distance matrix (B, N, N)
    dist_matrix = torch.linalg.vector_norm(batch.unsqueeze(2) - batch.unsqueeze(1), dim=-1)
    # Initialize tour and masks
    route = torch.zeros(B, N, dtype=torch.long, device=device)
    visited_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    # Select start at node (0) for all instances in the batch
    current_node = torch.zeros(B, dtype=torch.long, device=device)
    route[:, 0] = current_node
    visited_mask[torch.arange(B), current_node] = False
    # Build tour iteratively
    for i in range(1, N):
        # Get distances from the current node to all other nodes & mask out visited
        current_dists = dist_matrix[torch.arange(B), current_node]
        current_dists[~visited_mask] = MIN
        # Find the farthest unvisited node & update route and mask
        current_node = torch.argmax(current_dists, dim=1)
        route[:, i] = current_node
        visited_mask[torch.arange(B), current_node] = False
    # Reorder original coordinates based on the computed route
    return batch.gather(1, route.unsqueeze(-1).expand(-1, -1, 2))

def random_initial_tour(batch: torch.Tensor):
    """
    Creates a random initial tour for the TSP problem.
    Args:
        batch: (B, N, 2) - coordinates of nodes
    Returns:
        (B, N, 2) - nodes in random order
    """
    B, N, _ = batch.size()
    device = batch.device
    # Unique random permutation for each item in the batch
    random_values = torch.rand(B, N, device=device)
    # Get the indices that sort these values along the node dimension.
    random_indices = random_values.argsort(dim=1)
    return batch.gather(1, random_indices.unsqueeze(-1).expand(-1, -1, 2))

def get_initial_tours(batch: torch.Tensor, method: str):
    """
    Generate initial tours for a batch of TSP instances.
    Args:
        batch: (B, N, 2) - coordinates of nodes
        method: str - method to use for generating initial tours (greedy, random, farthest)
    Returns:
        (B, N, 2) - initial tours
    """
    if method == "greedy":
        return greedy_initial_tour(batch)
    elif method == "random":
        return random_initial_tour(batch)
    elif method == "farthest":
        return farthest_initial_tour(batch)
    else:
        raise ValueError(f"Unknown method: {method}")

def check_feasibility(solutions):
    """
    Checks if each tour in solutions visits every node exactly once.
    Args:
        solutions: (batch_size, size) tensor of node indices (permutations)
    Raises:
        AssertionError if any tour is infeasible.
    """
    batch_size, size = solutions.size()
    expected = torch.arange(size, device=solutions.device).view(1, -1).expand(batch_size, -1)
    is_feasible = (solutions.sort(dim=1)[0] == expected).all(dim=1)
    assert is_feasible.all(), "One or more tours do not visit all nodes exactly once."

def compute_euclidean_tour(tour):
    """
    Compute the tour length using euclidean with x & y coordinates
    Args:
        tour: (B, N, 2) - tour coordinates
    Returns:
        tour_length: (B,) - Euclidean tour length for each tour in the batch
    """
    assert tour.dim() == 3 and tour.size(-1) == 2, "Input must be of shape (B, N, 2)"
    # Shift coordinates by one in tour to get next node
    rolled_tour = torch.roll(tour, shifts=-1, dims=1)
    # Compute distances between consecutive nodes & return the sum
    segment_lengths = torch.linalg.vector_norm(tour - rolled_tour, ord=2, dim=-1)
    return segment_lengths.sum(dim=1)