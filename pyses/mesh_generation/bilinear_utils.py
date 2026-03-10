

def eval_bilinear_mapping(v0, v1,
                          v2, v3,
                          alpha,
                          beta):
  """
  Compute bilinear mapping for unstructured arrays of
  topological quadrilaterals in arbitrary cartesian dimension.

  Parameters
  ----------
  v0: Array[*Shape, Float]
    Upper left vertex, final index is cartesian dimension
  v1: Array[*Shape, Float]
    Upper right vertex, final index is cartesian dimension
  v2: Array[*Shape, Float]
    Bottom left vertex, final index is cartesian dimension
  v3: Array[*Shape, Float]
    Bottom right vertex, final index is cartesian dimension
  alpha: Float
    First coordinate within reference element
  beta: Float
    Second coordinate position within reference element

  Returns
  -------
  Array[*Shape, Float]
      Interpolated positions.

  Notes
  -----
  The reference element is assumed to be [-1, 1]^2.
  """
  #   v0---α---v1
  #   |    :    |
  #   |    β    |
  #   |    :    |
  #   v2---α---v3
  aprime = (alpha + 1) / 2
  bprime = (beta + 1) / 2
  top_point = aprime * v0 + (1 - aprime) * v1
  bottom_point = aprime * v2 + (1 - aprime) * v3
  return (bprime * top_point + (1 - bprime) * bottom_point)


def eval_bilinear_jacobian(v0, v1,
                           v2, v3,
                           alpha,
                           beta):
  """
  Compute jacobian of the bilinear mapping for unstructured arrays of
  topological quadrilaterals in arbitrary cartesian dimension.

  Parameters
  ----------
  v0: Array[*Shape, Float]
    Upper left vertex, final index is cartesian dimension
  v1: Array[*Shape, Float]
    Upper right vertex, final index is cartesian dimension
  v2: Array[*Shape, Float]
    Bottom left vertex, final index is cartesian dimension
  v3: Array[*Shape, Float]
    Bottom right vertex, final index is cartesian dimension
  alpha: Float
    First coordinate within reference element
  beta: Float
    Second coordinate within reference element

  Returns
  -------
  dphys_dalpha: Array[*Shape, Float]
    Derivative of each cartesian dimension
    w.r.t. the first coordinate on the reference element
  dphys_dalpha: Array[*Shape, Float]
    Derivative of each cartesian dimension
    w.r.t. the second coordinate on the reference element

  Notes
  -----
  The reference element is assumed to be [-1, 1]^2.
  """
  aprime = (alpha + 1) / 2
  bprime = (beta + 1) / 2
  dphys_dalpha = 1 / 2.0 * (bprime * (v0 - v1) + (1 - bprime) * (v2 - v3))
  dphys_dbeta = 1 / 2.0 * (aprime * v0 + (1 - aprime) * v1 - (aprime * v2 + (1 - aprime) * v3))
  return dphys_dalpha, dphys_dbeta
