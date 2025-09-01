import torch


def greedy_initial_tour(coords):
    """
    Creates an initial tour for the TSP problem using a greedy nearest neighbor heuristic.
    Args:
        coords: (batch_size, seq_length, 2)
    Returns: 
        (batch_size, seq_length, 2) - nodes in greedy order
    """
    batch_size, size, _ = coords.size()
    device = coords.device
    visited = torch.zeros(batch_size, size, dtype=torch.bool, device=device)
    route = torch.zeros(batch_size, size, dtype=torch.long, device=device)
    current = torch.zeros(batch_size, dtype=torch.long, device=device)
    route[:, 0] = current
    visited[:, 0] = True
    for i in range(1, size):
        last_coords = coords[torch.arange(batch_size), current].unsqueeze(1)
        dists = torch.norm(coords - last_coords, dim=2)
        dists[visited] = float('inf')
        next_node = dists.argmin(dim=1)
        route[:, i] = next_node
        visited[torch.arange(batch_size), next_node] = True
        current = next_node
    # Reorder coords according to route
    greedy_coords = coords.gather(1, route.unsqueeze(-1).expand(-1, -1, 2))
    return greedy_coords

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

def compute_soft_tour_length(soft_tour):
    """
    Computes the (differentiable) tour length for a soft tour.
    Args:
        soft_tour: (batch_size, seq_length, 2) - soft tour coordinates
    Returns:
        tour_length: (batch_size,) - differentiable tour length
    """
    # Compute pairwise distances between consecutive nodes
    diff = soft_tour[:, 1:, :] - soft_tour[:, :-1, :]  # (batch_size, seq_length-1, 2)
    segment_lengths = torch.norm(diff, dim=-1)         # (batch_size, seq_length-1)
    # Add distance from last to first to complete the tour
    last_to_first = torch.norm(soft_tour[:, 0, :] - soft_tour[:, -1, :], dim=-1)  # (batch_size,)
    tour_length = segment_lengths.sum(dim=1) + last_to_first
    return tour_length

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