import torch

# --- Setup ---
# Shape: (B=1, N=3, dim=2)
initial_tour = torch.tensor([[[10., 10.],
                              [20., 20.],
                              [30., 30.]]])
print("--- Initial Tour ---")
print(initial_tour.shape)
print(initial_tour)

# Shape: (B=1, N=3, N=3)
# P[i, j] = 1 means: move input node i to output position j
# We want: Node 0 -> Pos 1, Node 1 -> Pos 2, Node 2 -> Pos 0
permutation_matrix = torch.tensor([[[0., 1., 0.],
                                    [0., 0., 1.],
                                    [1., 0., 0.]]])
print("\n--- Permutation Matrix ---")
print(permutation_matrix.shape)
print(permutation_matrix)

# --- The Incorrect Way ---
incorrect_tour = torch.bmm(permutation_matrix, initial_tour)
print("\n--- Incorrect Result (without transpose) ---")
print(incorrect_tour.shape)
print(incorrect_tour)
print("This is a meaningless weighted average of coordinates.\n")


# --- The Correct Way ---
# Transpose swaps the node and position dimensions
transposed_permutation = permutation_matrix.transpose(1, 2)
correct_tour = torch.bmm(transposed_permutation, initial_tour)
print("\n--- Correct Result (WITH transpose) ---")
print(correct_tour.shape)
print(correct_tour)
print("This correctly reorders the nodes: Node 2, then Node 0, then Node 1.")