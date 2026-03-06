from ..config import np, DEBUG


def exodus_to_pysces_grid_corners(cart_coords, connect_map, element_permutation):
  # exodus uses 3 2
  #             0 1 ordering
  # we translate
  # note: we are using pysces edge conventions
  num_elem = element_permutation.shape[0]
  edges_forward = [(0, 1),
                  (0, 2),
                  (1, 3),
                  (2, 3)]

  edges_backward = [(x[1], x[0]) for x in edges_forward]

  exo_vert_to_pysces = [2, 3, 1, 0]


  node_elem_map = {}
  def wrap(node_idx):
    if node_idx not in node_elem_map.keys():
      node_elem_map[node_idx] = set()


  for elem_idx in range(connect_map.shape[0]):
    for node_idx in range(connect_map.shape[1]):
      node_num = connect_map[elem_idx, node_idx]
      wrap(node_num)
      node_elem_map[node_num].add((exo_vert_to_pysces[node_idx], element_permutation[elem_idx]))


  # here we make the assumption that at most one edge is shared between elements
  # (paired_elem_idx, paired_edge_idx, is_reversed) = edge_info[elem_idx, edge_idx, :]
  edge_info = -1 * np.ones((num_elem, 4, 3), dtype=np.int32)
  vert_pos = -1000 * np.ones((num_elem, 4, 3), dtype=np.float64)
  for elem_idx in range(connect_map.shape[0]):
    for node_idx in range(connect_map.shape[1]):
      elem_idx_loc = element_permutation[elem_idx]
      current_node = connect_map[elem_idx, node_idx]
      next_node = connect_map[elem_idx, (node_idx+1)%4]
      pysces_vert = exo_vert_to_pysces[node_idx]
      vert_pos[elem_idx_loc-1, pysces_vert, :] = cart_coords[:, current_node-1]
      pysces_edge = (exo_vert_to_pysces[node_idx],
                    exo_vert_to_pysces[(node_idx+1)%4])
      if pysces_edge in edges_backward:
        local_edge_idx = edges_backward.index(pysces_edge)
      else:
        local_edge_idx = edges_forward.index(pysces_edge)
      # just do this n^2 for now
      for (curr_node_pair_idx, curr_elem_pair_idx) in node_elem_map[current_node]:
        for (next_node_pair_idx, next_elem_pair_idx) in node_elem_map[next_node]:
          if curr_elem_pair_idx == next_elem_pair_idx and curr_elem_pair_idx != elem_idx_loc:
            edge_pair_elem_idx = curr_elem_pair_idx
            edge_pair = (curr_node_pair_idx, next_node_pair_idx)
            if edge_pair in edges_backward:
              remote_edge_idx = edges_backward.index(edge_pair)
            else:
              remote_edge_idx = edges_forward.index(edge_pair)
            remote_is_backwards = edge_pair in edges_backward
            local_is_backwards = pysces_edge in edges_backward
            edge_info[elem_idx_loc-1, local_edge_idx, 0] = edge_pair_elem_idx - 1
            edge_info[elem_idx_loc-1, local_edge_idx, 1] = remote_edge_idx
            edge_info[elem_idx_loc-1, local_edge_idx, 2] = remote_is_backwards ^ local_is_backwards
            break

  if DEBUG:
    assert np.all(edge_info >= 0)
    assert np.all(vert_pos > -999.0)

  for elem_idx in range(edge_info.shape[0]):
    for edge_idx_loc in range(edge_info.shape[1]):
      elem_idx_pair, edge_idx_pair, paired_edge_is_reversed = edge_info[elem_idx, edge_idx_loc, :]
      assert elem_idx_pair != elem_idx
      local_vert_1, local_vert_2 = edges_forward[edge_idx_loc]
      if paired_edge_is_reversed:
        remote_vert_1, remote_vert_2 = edges_backward[edge_idx_pair]
      else:
        remote_vert_1, remote_vert_2 = edges_forward[edge_idx_pair]
      if DEBUG:
        assert np.max(np.abs(vert_pos[elem_idx, local_vert_1, :] - vert_pos[elem_idx_pair, remote_vert_1, :])) < 1e-10
        assert np.max(np.abs(vert_pos[elem_idx, local_vert_2, :] - vert_pos[elem_idx_pair, remote_vert_2, :])) < 1e-10
  return vert_pos, edge_info
