import torch


def greedy_nearest_neighbor_tour(batch: torch.Tensor):
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

def greedy_farthest_neighbor_tour(batch: torch.Tensor):
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

def random_tour(batch: torch.Tensor):
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

def polar_tour(batch: torch.Tensor):
    """
    Creates an initial tour for the TSP problem using polar coordinates sorting around the centroid.
    Args:
        batch: (B, N, 2) - coordinates of nodes
    Returns:
        (B, N, 2) - nodes in polar sorted order
    """
    # Compute centroid for each instance and center the nodes
    centroid = batch.mean(dim=1, keepdim=True)  # (B, 1, 2)
    shifted_nodes = batch - centroid  # (B, N, 2)
    # Compute angles & sort indices based on angles
    angles = torch.atan2(shifted_nodes[:, :, 1], shifted_nodes[:, :, 0])  # (B, N)
    polar_indices = angles.argsort(dim=1)
    # Reorder original coordinates based on polar sorting
    return batch.gather(1, polar_indices.unsqueeze(-1).expand(-1, -1, 2))

def get_heuristic_tours(batch: torch.Tensor, method: str):
    """
    Generate tours for a batch of TSP instances using a specific heuristic.
    Args:
        batch: (B, N, 2) - coordinates of nodes
        method: str - method to use for generating tours
    Returns:
        (B, N, 2) - the newly ordered nodes according to the heuristic
    """
    if method == "greedy":
        return greedy_nearest_neighbor_tour(batch)
    elif method == "random":
        return random_tour(batch)
    elif method == "farthest":
        return greedy_farthest_neighbor_tour(batch)
    elif method == "polar":
        return polar_tour(batch)
    else:
        raise ValueError(f"Unknown method: {method}")

def check_feasibility(observation, solution):
    """
    Checks if observation and solution nodes are identical and that each node is visited exactly once.
    Args:
        observation: (B, N, 2) tensor of node coordinates
        solution: (B, N, 2) tensor of node coordinates in the proposed tour order
    Raises:
        AssertionError if any tour is infeasible.
    """
    # Sort both tensors along the node dimension for comparison
    # We sort by x-coordinate first, then y-coordinate (lexicographic sort)
    sol_sorted = solution[solution[:, :, 0].argsort(dim=1)]
    sol_sorted = sol_sorted[sol_sorted[:, :, 1].argsort(dim=1, stable=True)]
    
    obs_sorted = observation[observation[:, :, 0].argsort(dim=1)]
    obs_sorted = obs_sorted[obs_sorted[:, :, 1].argsort(dim=1, stable=True)]
    
    if not torch.allclose(obs_sorted, sol_sorted, rtol=1e-5, atol=1e-6):
        # Detach and move to CPU for safe printing, then raise with tensors included
        obs_print = observation.detach().cpu()
        sol_print = solution.detach().cpu()
        raise AssertionError(
            f"Solution nodes do not match observation nodes\nObservation:\n{obs_print}\nSolution:\n{sol_print}"
        )

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