.. asdf
Contributing
============

Coding Style
------------

The coding style in this codebase is somewhat odd in several ways. The code is designed 
so that JAX, Pytorch and Numpy can be used to do array calculations by changing a few configuration variables.
There are only a few functions that have code that branches based on which array provider is used. 
JAX uses a functional programming paradigm that is very close to how typical Fortran code is written, but does not
allow you to overwrite array memory after an array is initialized (`u[:, 0] = u_slice` will crash). However,
if your code avoids this, JAX can just-in-time compile your Python function to be highly performant on CPU and GPU
basically automatically. PyTorch's way of compiling code tends to work better on object-oriented code,
which is rather unfamiliar to older generations of atmospheric modelers. **Consequently, this codebase is written
to be performant in JAX**, which means that Numpy and PyTorch run slightly slower. Note: PyTorch and JAX will automatically
do shared-memory parallelism on the CPU, so care must be taken when comparing runtimes. 

How do we make an accessible, maintainable, friendly dycore?
-----------

* **Untested code is incomplete code.**
* **Undocumented code is incomplete code.**
* **Write functions, not subroutines.**
  * Doing the equivalent of `intent(out)` in a codebase that is mostly pure functions
is a bad idea. 
  * It will cause JAX to crash if you compile your function.
  Even if you aren't compiling your function, if will 
  * Avoid modifying input arrays wherever possible unless it is clearly marked in the docstring.
* Write code that prevents you from making stupid mistakes, even if it isn't documented.
  * Commenting that a function argument is a moist mixing ratio is vastly worse than naming the argument `[...]_moist_mr`
* Keep code flat, but use nested data
* Write clean code first, then optimize. GPUs are sitting idle because models can't even run on them. 
Programmer time is expensive, profiling is cheap. 
* Data structures should avoid containing physically inconsistent data 
  * If data can be used wrong, it will be used wrong.
  * Example: a dynamics state struct that contains tracer mixing ratios that are inconsistent with an intermediate value of mass coordinate `dpi`.
  * Do calculations in external variables, then re-wrap your data structure with physically consistent data.
* Functions returning intermediate quantities are not great. 
  * These intermediate quantities should be calculated in a separate function and passed in as an argument.
* Functions that return a single argument should be used like a 
* Data structures should not contain data fields that are not called for in the configuration
of the code you are using. 
  * For example: if you are not using tensor hyperviscosity, you should not pass a diffusion configuration structure
that could allow a tensor hypervisosity function to be called accidentally. 
  * If you are running hydrostatic code, your dynamics state struct should not contain fields like vertical velocity that are prognostic.
* Design code so that it is hard to separate tracer data from metadata (variable name, a field in the tracer struct, etc)
that expresses whether it is a dry mixing ratio (kg of tracer / kg of dry air), a moist mixing ratio (kg of tracer / kg of moist air),
or a mass quantity (kg of tracer per m^2 scaled by g).


Dependencies
------------

Dependencies are extremely bad if they increase the chance that an external change to a dependency
makes the model unusable unless someone puts time in to fix the code.  I want to design this code
so that it can go with minimal maintaintenance for around two years and still have a decent chance of
being installable in a few commands at the end of it. **If you want to add a dependency, you should
implement a (potentially less performant) fallback that can be switched on if that dependency breaks**.
On the other hand, if adding a (stable) dependency introduces a viable alternative to an existing dependency,
this is a good thing. 


Quickstarting your Fortran Port
-------------------------------
Typical Earth System Model (ESM) code that is written in Fortran 90/95 is typically 
very easy to naively port to python, just by going line by line and making the necessary replacements. 

Step 1: Unperformant Literal Translation
^^^^^^^
As your first step, just copy the file over and replace your functions with a literal translation into python
The most pressing are
* replacing array indexing with `()` to array indexing with `[]`
* changing 1-based indexing to 0-based indexing in individual arrays (e.g. zonal wind `v(i,j,1,k)` becomes `v[i,j,0,k]`)
* Changing do loops to for loops, and fixing the fact that the beginning index is zero-based, and the final index is exclusive in python.
For example, `do k=2,nlev` becomes `for k in range(1, nlev):`. 
Once you get this so that you can run the python functions with dummy inputs without crashing, 
you should test that your python function returns the same values as the Fortran version.
There are many ways to do this. The fastest is often just using print debugging, in 
which you run the Fortran code in situ (e.g., in CAM or EAM), print out the necessary fields for, say, a single spectral element
as it is passed into the function, and then intermediate values with indexes. Then pass those structs into your python 
function and go line-by-line making sure they are equivalent. You could also try f2py, but Earth System Model code is 
frequently so poorly modularized that this is really annoying to do.

Step 2: Write tests
^^^^^^^^^^^^^^^^^^^
Write tests that should be satisfied by the function that you're trying to port.
Sometimes these exist already, but a lot of ESM code just isn't. If you don't have tests you can just copy over,
here are a few tips on writing good tests:
* A smoke test (does this function get from beginning to end without crashing for a plausible input) is valuable in and of itself.
* Work out an example by hand, initialize array inputs manually, and check the result.
* Is there an egregiously slow way to compute what you're trying to compute, but it's really easy to code? 
Check if that returns the same result as your function for small inputs.
* Are there any conservation laws or positivity constraints that are satisfied? Figure out a way to generate random input variables
that don't have, e.g., negative mass or moisture, and check that the output satisfies those constraints for 100 random inputs.

Step 3: Write a more Numpy+Pythonic function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Things like for loops are quite slow in pure Python. JAX deals with for loops by unrolling them,
which means that for loops with, say, more than 100 iterations in the innermost loop are not ideal.
In this step, you will write a better version of your function in Python that will run much faster.

