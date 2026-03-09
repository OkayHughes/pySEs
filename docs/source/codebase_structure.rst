How is pySEs structured?
========================
Python is not very opinionated about how codebases are packaged. We are
very open to criticism that what we've done here is imperfect, and welcome suggestions for improvement.

* Source files and test files are separated, and are located in the `src` and `tests` subdirectories, respectively.
* The `src` directory is the root of the pysces package as it is installed by, e.g., `pip`.
* `.py` files in the base of the `src` directory (excepting `_config.py`) should expose friendly, clean
interfaces that model users who are not experts in dynamical cores can easily use.
* The `_config.py` file manages global model state relating to processor and parallelism configuration, and
exposes the `Backend` abstract base class, that lists the array functionality necessary to run pySEs 
