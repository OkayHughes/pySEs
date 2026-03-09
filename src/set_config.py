from argparse import ArgumentParser


if __name__ == "__main__":
  from config import write_config
  parser = ArgumentParser(prog='pysces_config',
                          description='Set computational configuration before running models.')
  parser.add_argument('-w', '--wrapper_type', default="none")
  parser.add_argument('-m', '--use_mpi', action='store_true', default=False)
  parser.add_argument('-g', '--use_gpu', action='store_true', default=False)
  parser.add_argument('-s', '--single_precision', action='store_true', default=False)
  parser.add_argument('-p', '--shard_cpu_count', default=1, type=int)

  args = parser.parse_args()
  wrapper_type = args.wrapper_type
  valid_wrappers = ["none", "jax"]
  shard_cpu_count = args.shard_cpu_count
  assert wrapper_type in valid_wrappers, f"Invalid wrapper type: {wrapper_type}, must be one of {valid_wrappers}"
  assert shard_cpu_count >= 1

  if shard_cpu_count > 1:
    assert wrapper_type == "jax", "Shard-based parallelism is only implemented for jax."
    assert not args.use_gpu, "Shard counts are automatic for GPUs."
  if wrapper_type == "jax":
    use_wrapper = True
  elif wrapper_type == "torch":
    use_wrapper = True
  else:
    use_wrapper = False
  write_config(debug=True,
               use_mpi=args.use_mpi,
               use_wrapper=use_wrapper,
               wrapper_type=wrapper_type,
               use_cpu=not args.use_gpu,
               use_double=not args.single_precision,
               shard_cpu_count=shard_cpu_count
               )
