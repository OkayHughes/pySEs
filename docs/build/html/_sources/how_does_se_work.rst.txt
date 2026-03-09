.. asdf
How does the Spectral Finite Element work?
==========================================

Let's say that you're mostly familiar with vectors of the form :math:`x \in \mathbb{R}^n`, and you're most familiar with 
formulating linear systems of equations in terms of :math:` A\mathbf{x} = \mathbf{b}`, where :math:`A \in \mathbb{R}^{n\times m}`, :math:`\mathbf{x} \in \mathbb{R}^m`,
and :math:`\mathbf{b} \in \mathbb{R}^n`. 

Throughout this work, we will work with what I call the "column sum" characterization of matrix multiplication, in which (working in :math:`\mathbb{R}^3` for concreteness)
math::
  \begin{pmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix} \begin{pmatrix}x_1 \\ x_2 \\ x_3 \end{pmatrix}
  &= x_1 \begin{pmatrix} 1 \\ 4 \\ 5 \end{pmatrix} + x_2 \begin{pmatrix} 2 \\ 5 \\ 8 \end{pmatrix} + x_3 \begin{pmatrix} 3 \\ 6 \\ 9 \end{pmatrix} \\
  &\stackrel{\textrm{algorithmic definition}}{=} \begin{pmatrix} x_1 + 2x_2 + 3x_3 \\ 4x_1 + 5x_2 + 6x_3 \\ 7x_1 + 8x_2 + 9x_3 \end{pmatrix}
Within this characterization of matrix multiplication, each column can be thought of in the "arrow" sense of vectors, and matrix multiplication
is simply scaling each column vector by a particular amount, then joining the scaled vectors end-to-end.

Linear structure
----------------

Why would we want more generality than this? 
^^^^^^^^^

"Linearity" typically looks like various extensions of the 1D linear system :math:`y=mx+b`. 
We want to be able to multiply by coefficients (e.g., :math:`m`) and add potential solutions (:math:`x, b`) together. 
You can verify using the laws of matrix multiplication that :math:`A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y}`,
and that :math:`A(a\mathbf{x} + \mathbf{y}) = aA\mathbf{x} + A\mathbf{y}` (:math:`a \in \mathbb{R}`). 
These equation show that vector addition and scalar multiplication, even if you've only encountered them as "joining arrows" or "scaling arrows" in `$$\mathbb{R}^n$$`, 
are "preserved", in some sense, by matrix multiplication.  

We know from introductory calculus that for (suitably not-evil) functions :math:`f, g : \mathbb{R} \to \mathbb{R}`, :math:`\int_0^1 f + g \intd{x} = \int_0^1 f \intd{x} + \int_0^1 g \intd{x}`, and that 
:math:` \int_0^1 af + g \intd{x} = a\int_0^1 f \intd{x} + \int_0^1 g \intd{x}`. 
Likewise, if :math:` f, g ` are continuously first-differentiable :math:` \der{}{x} \left[af + g \right] = a\der{f}{x} + \der{g}{x} `.
In a sense that can be made extremely mathematically precise, the operations of differentiation and integration can be used to formulate linear equations (but which are MUCH harder to solve/characterize than systems in :math:`\mathbb{R}^n`, because they are infinite dimensional). 
However, in their most general form, there isn't a very natural way to write differentiation and integration as a matrix (integration can sometimes be thought of as an infinite-dimensional matrix, in a limiting sense, but differentiation in infinite dimensions absolutely cannot be thought of this way). 
This leads us to the question "what properties do matrices and more general linear operations share". 
This is one reason to define "linear operators", of which matrices, integration, and differentiation are examples. In finite dimensions, operators always have a matrix representation.  

Let's continue this analogy even further: suppose I have the system of equations
math::
  A\mathbf{x} = 
  \begin{pmatrix}
  1& 0 & 0 \\ 
  0 & 1 & 0 \\
  0 & 0 & 0
  \end{pmatrix} \mathbf{x} = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}
:math:` \mathbf{x} = (1, 1, 0)^\top$$` is a solution, but so is `$$ \mathbf{x} = (1, 1, \textrm{one hkjdillion})^\top`. 
It turns out that this is because for :math:` \mathbf{n} = (0, 0, 1)^\top$$`,  `$$ A\mathbf{n} = 0`, meaning that for any scalar :math:` a \in \mathbb{R}`, :math:`A(a\mathbf{n}) = aA\mathbf{n} = 0a = 0`. 
Therefore if we have a solution to :math:`A\mathbf{x} = \mathbf{b}`, then :math:`A (\mathbf{x} + a\mathbf{n}) = A\mathbf{x} + aA\mathbf{n} = A\mathbf{x} + 0 = b`, so :math:`\mathbf{x} + a\mathbf{n}` is also a solution. Therefore this system of linear equations has an infinite number of solutions.
Suppose your calculus teacher asked you to find functions :math:`f` that satisfy :math:` \der{f}{x} = x^2`. 
You would immediately go "well, there are infinitely many such :math:`f`, because :math:` \der{}{x}\left[f + C \right] = \der{f}{x}` for any constant :math:`C`". 
Therefore for any :math:`f` that is a solution to that equation, :math:`f + C ` is also a solution. In both cases, non-uniqueness of solutions to a linear equation
can be derived from the fact that the linear operators (:math:`A, \der{}{x}`) have certain non-zero vectors that they squash (the space of such vectors is called the nullspace). 
By talking about linear operators more generally, we can use the tools of linear algebra to examine the properties of operators like :math:`\der{}{x}`.


This means that there should be some analogy between a summation :math:`(0, 0, 1)^\top + (0, 1, 0)^\top = (0, 1, 1)^\top ` and :math:` f + g`, right? 
Exactly, but you've already been doing this since you were 13. 
If we let :math:`f = x^3` and :math:`g = x^2`, :math:` f + 2g = x^3 + 2x^2`.  
Put formally, if you have two functions :math:`f, g : \mathbb{R} \to \mathbb{R}`, and :math:`a \in \mathbb{R}`, :math:`(af +g)(x) = af(x) + g(x)`. 
For a moment, let's consider polynomials of a fixed maximum degree, e.g. 1D polynomials that contain no power higher than `$$x^3$$`. 
These take the form of :math:` ax^3 + bx^2 + cx + d`. Hmm.
What if I make the definition :math:` \mathbf{e}_0 = 1, \mathbf{e}_1 = x, \mathbf{e}_2 = x^2, \mathbf{e}_3 = x^3 `. 
Then it seems that for a cubic polynomial :math:` f `, I can write it as :math:` f = d\mathbf{e}_0 + c \mathbf{e}_1 + b \mathbf{e}_2 + a \mathbf{e}_3`. 
Recall that in :math:`\mathbb{R}^n`, we define the basis vectors as :math:` \mathbf{e}_i = (0, \ldots, \stackrel{i}{1}, \ldots, 0)^\top `, so a vector :math:` \mathbf{x} = (x_1, x_2, \ldots, x_n)^\top` can be written as :math:` \mathbf{x} = \sum_i x_i \mathbf{e}_i`. 
If we let the space :math:`V^3_1(\mathbb{R})` be the set of all cubic polynomials in one real variable, it turns out that this is a four-dimensional vector space that is mathematically indistinguishable from (isomorphic to) :math:` \mathbb{R}^4 `. We will continue this analogy in the next section.


As a cautionary note, you might look at the last example and say "well, I looked at the problem in the last section and started solving it by integrating both sides of the equation, so shouldn't there be a way to formalize that differentiation  and integration are analogous to :math:` A ` and :math:`A^{-1}`. 
The answer is yes, in many contexts there are ways to formalize this (these are typically called "the fundamental theorem of calculus" for whatever structure you're working with, :math:`L^p` spaces, general banach spaces, sobolev spaces, etc). 
However, if you aren't adequately mathematically precise about how you do this, you produce mathematical statements that are either wrong or so ambiguous that they may as well be wrong. 
I've personally wasted weeks of my life where physicists somehow got this crap published and I had to figure out what they actually meant (this gets especially hairy when you're dealing with PDEs with boundary conditions). 
If you aren't planning to spend a year learning intro graduate analysis, don't DIY this. Ask someone to help you.



A little finite dimensional example 
^^^^^^^^^^^^^^^^^

Let's return to :math:` V^3_1(\mathbb{R}) `, the space of cubic polynomials in one real variable, (I forget the standard notation for this). 
The identity :math:` \der{}{x} \left[x^n \right] = nx^{n-1}`
means that if we choose the monomial basis :math:` \mathbf{e}_n = x^{n-1}, n=1,\ldots,4` and write polynomials as :math:` \mathbf{f} = \sum_n f_n \mathbf{e}_n `, then we can ask: is there a matrix :math:` D ` such that if we associate a polynomial  :math:`f ` to its representation :math:` \mathbf{f} \in \mathbf{R}^4 `, then the  :math:` D\mathbf{f} ` is equivalent to the representation of :math:`f'` as a vector? 
Yes. It looks like 
math::
  D \equiv \begin{pmatrix}
  0 & C & 0 & 0 \\
  0 & 0 & 2 & 0 \\
  0 & 0 & 0 & 3 \\
  0 & 0 & 0 & 0   
  \end{pmatrix}
some good exercises: 
* Take a deep breath, don't panic like I did when I went through this example when I was in college. If you took high school math and understood it, you can do this!
* how would you encode our monomial basis, e.g. :math:` e_2 = x^2` (recall that vectors index top to bottom) ? How does the matrix act on this vector? 
* Use the information from the last question to determine the value of :math:`C`. 
* Verify that the matrix behaves how you would expect it to when calculating :math:`D(f + g) = Df + Dg` for :math:`f = 6x^3 + 3`, :math:`g = 2x`. 
* this matrix is not invertible. Why? How does this relate to your understanding of calculus?
* How would you extend this matrix for :math:`V^4_1(\mathbb{R})`, which is the space of at-most-quartic polynomials in one real variable.
* Hard exercise: What would the matrix look like for :math:` V^1_2(\mathbb{R})`, the space of at-most-linear polynomials in two real variables.
  * Step 1: What is the dimension of this space?
  * Step 2: What does your monomial basis look like?
  * Step 3: Does it actually matter what order you put the monomials in?
  * Step 4: Write down the matrix. 
  
Quadrature
""""""""

Let's now work over :math:` V_1^2(\mathbb{R})` for brevity. 
Let's say I want to compute :math:` \int_I f \intd{x} = \int_I \sum_n f_n x^n \intd{x}` for some interval :math:`I` (you can assume :math:` I = [0,1]` without losing any generality).  
Well, by expanding we find that :math:` \int_I \sum_n f_n x^n \intd{x} = \sum_n f_n \int_I x^n \intd{x}`. 
Wait, :math:` \int_I x^n \intd{x}` doesn't seem to depend on my specific polynomial, :math:`f`, at all! 
In fact, it doesn't, so one can compute :math:`w_n = \int_I x^n \intd{x}`. 
Suddenly, we can compute :math:`\int_I f \intd{x} = \sum_n f_n w_n = \langle \mathbf{f}, \mathbf{w} \rangle ` and this is an analytic integral that contains no numerical approximation whatsoever. 
:math:` \langle \cdot, \cdot \rangle ` is the standard inner (dot) product on :math:`\mathbb{R}^3` (see below).
This is one way in which integration can be represented as a linear operator (in this case, the matrix representation of :math:`w_n` would be the "row vector" :math:` (w_1, \ldots, w_n) `.

You actually do need to learn what a basis is
--------------------

First we need to talk about what bases are. 
In :math:`\mathbb{R}^n`, a basis is a set of :math:`n` vectors :math:`\mathbf{b}_k` such that any :math:`\mathbf{x} \in \mathbb{R}^n` can be written as :math:`\sum_k b_k \mathbf{b}_k = \mathbf{x}` (where :math:`b_k \in \mathbb{R}` in exactly one way). 
It can be shown that a basis must contain exactly :math:`n` vectors. 
Any more than that, and there will be multiple :math:`b_k` that can be used to reconstruct :math:`\mathbf{x}`.  
Any fewer and there will be vectors :math:`x` that cannot be reconstructed by any choice of :math:`b_k`. 
To see why this is, imagine that I choose :math:`\mathbf{b}_1 = (1, 0, 0)^\top `, :math:`\mathbf{b}_2 = (0, 1, 0)^\top`, and :math:`\mathbf{b}_3 = (1, 1, 0)^\top`. 
If we let each entry of the vector be the dimensions :math:`x, y, z`, respectively, then we see that the first two vectors can be used to reconstruct (span) the plane :math:` z = 0`. 
But our third vector lies in that plane. 
That means that any vector :math:`\mathbf{x} = (x_1, x_2, x_3)` with :math:`x_3 \neq 0` lies out of reach of our :math:`\mathbf{b}_n`. 
Any failure of :math:`n` vectors to be a basis for :math:`\mathbb{R}^n` fails in exactly this way, which can be formalized. 
If you stack your :math:`\mathbf{b}_k` as columns of a matrix, this equivalent to that matrix being invertible. 
Indeed, if you have a vector :math:` \mathbf{x} = \sum x_k \mathbf{e}_k`, with :math:`\mathbf{e}_k = (0, \ldots, \stackrel{k}{1}, \ldots, 0)` the "standard basis" on :math:`\mathbb{R}^n`,
then solving the system 
math::
  \begin{bmatrix} \mathbf{b}_1 & \ldots & \mathbf{b}_k \end{bmatrix} \mathbf{x}_\mathbf{b} = \mathbf{x}_\mathbf{e}
finds :math:`\mathbf{x}_\mathbf{b} = (b_1, \ldots, b_n)` that reconstruct :math:`\mathbf{x}` in the :math:`\mathbf{b}_k` basis.


Ok, let's see how matrices and bases are related using the concrete example of the polynomial space above.
Let's suppose we have :math:`f = \sum_{k=1}^4 f_k x^{k-1}`, and that I want to characterize the :math:`\der{}{x}` operator.
Then we find 
math::
  \der{f}{x} = \der{}{x} \left[ \sum_{k=1}^4 f_k x^{k-1} \right]  = \sum_{k=1}^4 f_k \der{}{x} \left[ x^{k-1} \right] = \sum_{k=1}^4 f_k (k-1)x^{k-2}.
If we make the association we made earlier, with :math:` x^{0} \simeq (1, 0, 0, 0)^\top, x \simeq (0, 1, 0, 0)^\top, x^2 \simeq (0, 0, 1, 0)^\top, x^3 \simeq (0, 0, 0, 1)^\top `
then we find that the matrix representation of our differentiation operator, :math:`D` should satisfy
math::
    \der{f}{x} &= f_1 \der{}{x} \left[1 \right] + f_2 \der{}{x} \left[x \right] + f_3\cdot \der{}{x} \left[x^2 \right] + f_4 \cdot \der{}{x}\left[ x^3\right]   \\
    &= f_1 (0) + f_2 \cdot 1  + f_3\cdot 2 x + f_4 \cdot 3x^2   \\
    &\simeq f_1 \cdot 0 + f_2 \cdot 1 \begin{pmatrix}1 \\ 0 \\ 0 \\ 0 \end{pmatrix} + f_3 \cdot 2 \begin{pmatrix}0 \\ 1 \\ 0 \\ 0 \end{pmatrix} + f_4 \cdot 3 \begin{pmatrix}0 \\ 0 \\ 1 \\ 0 \end{pmatrix} \\
    &= f_1 \begin{pmatrix}0 \\ 0 \\ 0 \\ 0 \end{pmatrix} + f_2 \begin{pmatrix}1 \\ 0 \\ 0 \\ 0 \end{pmatrix} + f_3 \begin{pmatrix}0 \\ 2 \\ 0 \\ 0 \end{pmatrix} + f_4 \begin{pmatrix}0 \\ 0 \\ 3 \\ 0 \end{pmatrix}  \\
    &= \begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \\ 0 & 0 & 0 & 0 \end{pmatrix}\begin{pmatrix}f_1 \\ f_2 \\ f_3 \\ f_4 \end{pmatrix} 
which means that we have found a matrix representation of :math:`\der{}{x}` just by examining how it acts on a basis of :math:` V_1^3(\mathbb{R})`. 

Some exercises:
* The polynomials :math:` -1, x, -x^2, x^3 ` also form a basis of :math:` V_1^3(\mathbb{R})`. 
    * Letting the standard monomial basis be identified with :math:` \mathbf{e}_k`, and this new basis be :math:` \mathbf{b}_l`, write each new basis vector in terms of 
    the "standard" monomial basis, i.e. find :math:`e_{k, l}` so that :math:` \mathbf{b}_k = \sum_{l=1}^4 e_{k, l} \mathbf{e}_l`. Hint: :math:` e_{2,2} = 1`, :math:`e_{2, \{1, 3, 4\}} = 0`. 
    * Use these coefficients to form the matrix equation :math:` \begin{bmatrix} \mathbf{b}_1 ~ \ldots ~ \mathbf{b}_k \end{bmatrix} \mathbf{x}_\mathbf{b} = \mathbf{x}_\mathbf{e}.` How
    does this equation show that :math:`-1, x, -x^2, x^3` form a basis? (Hint: use invertibility and the determinant).
    * Redo the process above to derive the derivative operator in the :math:`\mathbf{b}_k` basis. Show that the derivative of a constant polynomial is still zero.

Note: this exercise shows that the matrix representation of :math:` \der{}{x} ` is different for different bases. However, it can be shown that features of the linear operator :math:`\der{}{x}` like
the nullspace do not depend on this choice of basis. As a result, while representation of linear operators in a basis (i.e., as a matrix) are helpful for calculations (and certain bases can make it easy to read off properties of your linear operator from the matrix itself), most properties of :math:`A` do not depend on what basis you write it in. I have also glossed over what happens when the dimensions of the domain and range of your linear operator are not equal, in which case you must use separate bases of the domain and range to find the coefficients of your matrix.

.. <!--Let's pretend that we have a linear operator :math:` A ` that behaves "like a matrix" (most concisely, for :math:` a, b \in \mathbb{R}`, :math:`\mathbf{x}, \mathbf{y} \in \mathbb{R}^n \textrm{ (maybe even } V)`, :math:` A(a\mathbf{x} + b \mathbf{y}) = aA\mathbf{x} + bA\mathbf{y} `) and that we have a basis :math:` \mathbf{b}_k`. 
.. First, let's write :math:` \mathbf{x} = \sum_l e_{l} \mathbf{e}_l` (:math:`e_{l}` are just the entries of :math:`\mathbf{x}` in a column vector!). What happens when we look at how :math:`A` behaves on :math:`\mathbf{x}`
.. math::
..   A\left(\mathbf{x} \right) &= A\left(\sum_l e_{l} \mathbf{e}_l \right) \\
..     &= \sum_l e_{l} A\left( \mathbf{e}_l \right) \\
..     &= \sum_l e_{l} \mathbf{a}_l,
.. BUT: :math:` \mathbf{a}_l` is the image of the basis vector :math:` \mathbf{e}_l` under :math:`A`. If :math:`A` were a matrix, this would be the :math:`l`th column of the matrix (do the multiplication yourself to
.. convince yourself this is right).   -->


Inner products, duals
----------

An inner product is an additional operation `$$\langle \cdot, \cdot \rangle $$` that one can add to a vector space. 
It must satisfy certain [requirements](https://en.wikipedia.org/wiki/Inner_product_space). 
It is a generalization of the dot product on `$$\mathbb{R}^n$$`, `$$ \langle \mathbf{x}, \mathbf{y} \rangle = \sum_k x_k y_k $$`. 
The aforementioned requirements are the minimal set of constraints required to guarantee that the inner product you defined on your vector space `$$V$$`, gets you the properties that make the dot product useful. The most concise way to state these properties are that for  `$$x, y, z \in V $$`, `$$a, b \in \mathbb{R} $$`, `$$\langle x, y \rangle = \langle y, x \rangle $$` (this changes if you have a complex vector space!), `$$ \langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, x \rangle $$`, and `$$ \langle x, x \rangle > 0, \langle x, x \rangle = 0 \iff x = 0 $$`. 

## Why are orthogonal bases good?
Bases do not require an inner product to be defined and analyzed. However, if we have an inner product, we can define a condition for a basis to be particularly nice.
The standard basis `$$ \mathbf{e}_k$$` satisfies `$$$ \langle \mathbf{e}_i, \mathbf{e}_j \rangle = \begin{cases} 1 \textrm{ if } i = j \\ 0 \textrm{ otherwise} \end{cases} $$$`,
meaning that for distinct `$$ \mathbf{e}_i, \mathbf{e}_j$$`, the vectors are at 90º to each other. A basis that satisfies this latter property alone is called "orthogonal"; if, additionally, `$$\langle e_i, e_i \rangle = 1$$`, then the basis is called "orthonormal". 

Recall that for a general basis `$$\mathbf{b}_k $$`, solving for the coefficients `$$ b_k $$` that reconstruct a vector `$$\mathbf{x}$$` as `$$ \mathbf{x} = \sum b_k \mathbf{b_k}$$` requires a full linear solve. 
However, let `$$ \mathbf{e}_l $$` be an orthonormal basis. First, suppose we do the full solve to reconstruct `$$ \mathbf{x} = \sum e_l \mathbf{e}_l$$`.  
Then we can calculate 
`$$$
\langle x, \mathbf{e}_k \rangle = \langle \sum e_l \mathbf{e}_l, \mathbf{e}_k \rangle = \sum e_l \langle \mathbf{e}_l, \mathbf{e}_k \rangle = \sum e_l \begin{cases}1 \textrm{ if } k = l \\ 0 \textrm{ otherwise}\end{cases} = e_k.
$$$`
What this means is, the coefficient `$$ e_k $$` in the sum above can be calculated simply from `$$ \langle x, e_k \rangle $$`. You don't have to solve a full system. 


A concrete example from function approximation: The construction of the Legendre polynomials:
^^^^^^^^^^^^^^^^


Let's return to the example of polynomial quadrature above. It turns out that for, e.g., `$$ f, g \in V^3_1 (\mathbb{R})$$`, making the definition `$$ \langle f, g \rangle \equiv \int_I fg \intd{x} $$` satisfies the requirements of an inner product! Starting with the convention `$$\mathbf{l}_1 = 1$$`, and requiring that the polynomial `$$\mathbf{l}_{k+1}$$` satisfy `$$ \langle \mathbf{l}_{k+1}, \mathbf{l}_l \rangle = 0$$` for `$$ l \leq k $$` (this is analogous to gram-schmidt, if you've encountered that), you generate a sequence of polynomials that are an orthogonal (easily made orthonormal) basis for `$$V_1^n(\mathbb{R})$$`. Using the standard (unweighted) inner product `$$ \int fg \intd{x} $$`, this generates the Legendre polynomials. 
## The adjoint, and bilinear mappings
While the adjoint of an operator can be defined without any reference to an inner product, it is much easier to talk about them when we do have an inner product.

Let's first work in `$$\mathbb{R}^n$$`. Suppose we have some linear operator `$$A : \mathbb{R}^n \mapsto \mathbb{R}^n $$`, which can be written in the standard basis with entries `$$ A_{i,j}$$`.  You can extract this matrix representation by computing
`$$$
 \langle e_j, A(e_i) \rangle = \mathbf{e}_j^\top A \mathbf{e}_i  = A_{i,j}.
$$$`

This gives some indication that all of the information contained in the linear operator `$$ A $$` can be extracted by the construction `$$ b_A(x, y) = \langle x, Ay \rangle $$`. One can verify that `$$b_A(x, y)$$` is linear in each argument, and returns a real number (this is called a bilinear form). For reasons that we will see soon, it will be worth asking the question, is there a linear operator `$$A'$$` such that`$$\langle x, A(y) \rangle = \langle A'(x), y \rangle   $$`? Well, let's see if we can figure it out by working in a basis:
`$$$
\begin{align*}
     \langle e_k, A(e_i) \rangle &= \langle A'(e_j), e_i \rangle \\
       &= \langle e_i, A'(e_j) \rangle,
\end{align*}
$$$`
but we know that the final line extracts the entries of `$$A'$$` in the standard basis, therefore we see `$$ A_{i,j} = A'_{j,i}$$`, or, instead `$$ A' = A^\top $$`. This shows
that for a real vector space, the adjoint of an operator can be characterized by the transpose of a matrix representation.
## Weak form of an equation

The Galerkin method first derives from the fact that differential equations can often be rewritten equivalently 
by using integration by parts to transfer differential operators into integral ones. We will discuss a prototypical example, `$$ \nabla^2 u = f $$`
We can rewrite this as `$$ \nabla \cdot \nabla u = f$$`, and then if we multiply by a suitable function `$$v$$` and integrate both sides of the equation, we get
`$$$
\begin{align*}
    \int fv \intd{x} &= \int v\nabla \cdot \nabla u \\
        &= 0 - \int \nabla v \cdot \nabla u \intd{x} 
\end{align*}
$$$`

For technical reasons, there are solutions `$$u$$` to this rewritten (weak form) equation that do not satisfy `$$ \nabla^2 u = f$$` (strong form), but any `$$u$$` that satisfies the 
strong form satisfy the weak form. 

If are vector-valued functions `$$ \mathbf{q}, \mathbf{r} : \mathbb{R}^n \to \mathbb{R}^n$$`, then `$$ \langle \mathbf{q}, \mathbf{r} \rangle  \equiv \int \mathbf{q} \cdot \mathbf{r} \intd{x}$$` is an inner product on that space of functions. As one final jargon explanation, `$$ \tilde{f}(v) = \int fv \intd{x} $$` is linear in `$$v$$` and returns a real value. We refer to functions like `$$\tilde{f}$$` as "linear functionals".

Therefore, we can rewrite our strong equation `$$ \nabla^2 u = f$$` as `$$ \int \nabla v \cdot \nabla u \intd{x} \equiv \langle \nabla u, \nabla v \rangle  = \tilde{f}(v) \equiv \int fv \intd{x} $$`.
Note that `$$ a(u, v) = \langle \nabla u, \nabla v \rangle $$` is a bilinear form, and indeed `$$ a(v, u) = \langle \nabla u, \nabla v \rangle = \langle \nabla v, \nabla a \rangle = a(u, v)$$` so `$$a$$` 
is symmetric.  

Galerkin
-------

Galerkin methods start with assuming that `$$u, v$$` are elements of some function space `$$V$$` that's typically infinite dimensional (frequently `$$L^2$$` or `$$H^1$$`) complete normed vector spaces.
Bilinear forms, like `$$a(u,v)$$` can be defined without a defined inner product, but the inner product allows you to construct bilinear forms very naturally. 
Let `$$a(u, v) $$` be a bilinear form and a linear functional `$$f$$`, and say we want to solve the equation `$$a(u,v) = f(v)$$`. Many PDEs can be rewritten in this form (the general idea here
is very naturally extended to forms that are not bilinear, but which are linearized as part of newton's method). If we know that this problem is well posed (i.e. it has a unique solution),
then we want a sequence of finite-dimensional problems whose solutions `$$u_n$$` will approach (converge to) the infinite-dimensional solution `$$u$$` as the size of our
approximation of the function space increases.
## Bobunov-Galerkin
For simplicity let `$$V$$` be a nice space of functions of two real variables. Let `$$V_2^n$$` be, e.g., the space of at-most-n-degree polynomials in two real variables. `$$V_2^n \subset V$$` is called a subspace (a subset that is a vector space in which vector addition and scalar multiplication agree on `$$ V_2^n \cap V$$`). 
The finite-dimensional approximation of our problem then looks like: find `$$u_n$$` such that  for all`$$ v \in V_2^n$$`, `$$ a(u_n, v) = f(v)$$`.
In finite dimensions, it suffices to let `$$v_1, \ldots, v_n$$` be a basis. Let us use the concrete problem above to see what this looks like. Let `$$v_k$$` be some nice polynomial basis.
`$$$
\begin{align*}
  \int f v_k \intd{x} &= \int \nabla u \cdot \nabla v_k \intd{x} \\
    &= \int  \sum_l \left(u_l \nabla  v_l \right)  \cdot \nabla v_k \intd{x}
\end{align*}
$$$` 

If we let `$$A$$` be the matrix with entries `$$A_{k, l} = \langle \nabla v_l, \nabla v_k \rangle $$`, and `$$ \mathbf{b}$$` be the vector with entries `$$ b_k = \int f v_k \intd{x} $$`,
then the above equation becomes
`$$$
    A \mathbf u = \mathbf{b},
$$$`
which can be solved however you like! It can be shown that for large classes of `$$a(u, v)$$`, `$$\lim_{n\to\infty}  \| u_n - u \|_V = 0$$`, i.e., that 
our finite dimensional approximations will approach the infinite dimensional solution.

Petrov-Galerkin
^^^^^^^^

Test functions `$$w_n$$` don't have to live in the same vector space as trial functions `$$u_n = \sum_k u_k v_k $$`. This situation
arises a lot when you have boundary conditions, and also equations with operators of odd order. 
You can also get faster numerical convergence by choosing `$$w_n$$` to be a different function space, e.g., by
weighting test functions upstream of the flow.

In this case, the galerkin problem looks like letting `$$ V_n \subset V$$`, `$$W_n \subset W$$`, and satisfying (in infinite dimensions)
`$$$
    a(u, w) = f(w) \forall w \in W 
$$$`
and discretely
`$$$
    a(u, w_n) = f(w_n) \textrm{ for } w = w_1, \ldots, w_n
$$$`
which can be used to write a concrete linear system 
just as above.


An example, concretely:
---------------

In this section we go through building a nodal SEM for the linear advection equation on the periodic unit interval `$$\Omega = [-1, 1]$$`. Note that
since we're using periodic boundary conditions, `$$\partial \Omega = \empty $$` 
In the continuum this equation reads as `$$ \pder{u}{t} +  \pder{}{x} [F(u)] = 0 $$` in the strong form. 
In the weak form, let `$$\psi $$` be a sufficiently regular test function. Then multiply and integrate to get
`$$$
\begin{align*}
    0 &= \int_\Omega \psi \pder{u}{t} \intd{x} + \int_\Omega \psi \pder{}{x} F(u)   \intd{x} \\
    &= \int_\Omega \psi \pder{u}{t} \intd{x} + \stackrel{=0}{\left[\psi F(u) \right]_{\partial \Omega}} - \int \pder{\psi}{x} F(u) \intd{x} \\
    &= \int_\Omega \psi \pder{u}{t} \intd{x} - \int \pder{\psi}{x} F(u) \intd{x} \\
\end{align*}
$$$` 
Note that `$$F$$` might be a nonlinear operator in the general case, but in all cases we see that the first term is a linear functional in `$$\pder{u}{t} $$`. For, e.g., linear advection with `$$F(u) = a(x) u$$`, 
we immediately see that the second term becomes a bilinear form in `$$\psi, u$$`. There isn't an inherent reason that `$$u, \psi$$` need to lie in exactly the same function space, so this
looks like a Petrov-Galerkin method might apply (indeed, for the paper we're discussing, we end up using a Bobunov-Galerkin method). We're going to let `$$u, \psi $$` live in the same function space for the moment.

Let `$$u, \psi \in V$$` be functions (i.e., infinite dimensional vectors). Let's apply a Galerkin method and see what our equation looks like: let `$$ \psi_j  \in V_n$$` be basis functions. For the moment, we
choose them without specifying any kind of orthogonality.

 Expand `$$ u_n$$` in this basis as `$$ u_n = \sum_j a_j \psi_j$$`. 
We do immediately see a potential complication: we've declared that our solutions must be periodic, so `$$u(-1) = u(1)$$`.  
You could derive a finite element method with `$$\psi_j$$` non-periodic by expanding `$$u_n$$` in a separate basis, but for the SEM this is highly undesirable.
It turns out that it's not particularly hard to construct periodic basis functions, so we assume `$$\psi_j$$` are periodic.
 The above equation reads
`$$$
\begin{align*}
 0 &= \int \psi_k \pder{}{t} \sum_j a_j \psi_j \intd{x} - \int \pder{\psi_k}{x} F(u_n) \intd{x} \\
   &= \sum_j \pder{a_j}{t} \int\psi_k\psi_j \intd{x} - \int \pder{\psi_k}{x} F(u_n) \intd{x}.
\end{align*}
$$$`
The term `$$\int \psi_k \psi_j \intd{x}$$` is what is termed the "mass matrix" in finite element theory. 
In the above form, the second term turns out to be a vector of length `$$n$$`, and the first term is a `$$n\times n$$` matrix.
Deriving your time tendency requires inverting a matrix, which is a computationally expensive operation (`$$>\mathcal{O}(n^2)$$`).
What if we chose our basis to make this better? This is what the "spectral" in spectral finite elements refers to.
If we let `$$ \psi_k$$` be orthogonal with respect to the inner product `$$\int fg \intd{x}$$`, then the mass matrix `$$ \int \psi_k \psi_j \intd{x} = \delta_{j,k}$$`, so the equation simplifies to 
`$$$
   \pder{a_k}{t} = \int \pder{\psi_k}{x} F(u_n) \intd{x}.
$$$`

This appears to be simpler. However: these equations still involve integrals. At this point, the orthogonality of `$$\psi_j$$` has only been specified with 
respect to the infinite-dimensional inner product. At the end of the day, if we have `$$n$$` basis functions, you can mathematically show that the vector subspace `$$V_n \subset V$$` is mathematically equivalent (isometric) 
to `$$\mathbb{R}^n$$` with the standard inner product. Therefore, we should expect that the operation `$$ \int fg \intd{x}$$` should take `$$n$$` operations. 
There are an unlimited number of ways to approximate `$$\int fg \intd{x}$$` with a finite number of operations. 

There are two pieces of jargon that you'll encounter in the SEM literature: the "modal" basis and the "nodal" basis. The exposition we've given so far 
makes the "modal" basis choice pretty easy to construct. In a modal method (I'm being slightly imprecise here), if we expand `$$ f = \sum f_k \psi_k, g = \sum g_j \psi_j$$`, 
with `$$ f \sim (f_1, \ldots, f_n) \in \mathbb{R}^n$$`, then `$$ \int fg \intd{x} = \int \sum_k \sum_j f_k g_j \psi_k \psi_j \intd{x} = \sum_k \sum_j f_k g_j \delta_{k,j} = \sum_k f_k g_k $$`.
This is precisely the way to map the vector space spanned by `$$\psi_j$$` to `$$\mathbb{R}^n$$` so that the inner product is just the dot product. This seems very mathematically simple,
so why would you choose anything other than this basis? Well, the problem turns out to be interpreting quantities like `$$ F(u_n) \pder{\psi}{k}$$` in 
our problem above. If `$$F$$` is non-linear, there is no guarantee that the result `$$ F(u_n(x))$$` will lie in `$$ V_n$$` even when `$$u_n$$` does (even in the linear case,
there is no reason to expect this to be the case).  `$$F(u_n(x))$$` must be squashed (projected) back onto `$$V_n$$` in order for the discrete inner product to make sense. 

Notice that in this section, we haven't talked about evaluation of our functions at particular points. In most problems, it is much easier to calculate `$$F$$` given `$$u_n(x)$$` than
given coefficients of `$$u_n$$` in the `$$\psi_k$$` basis. The broad concept  of a "nodal" basis is that you can choose a 
different polynomial basis `$$\xi_k$$`, and a different method of mapping these polynomials onto `$$\mathbb{R}^n$$`, then there are a set of points `$$x_k$$`
such that the best approximation of `$$ F(u_n) = \sum_k F(u_n(x_k)) \xi_k $$`. In the next section I'll explain how that works.



Gauss quadrature: why is it special?
^^^^^^^^^

For the second part of our derivation, recall that for any set of `$$n$$` function evaluations `$$(x_k, f(x_k)$$` on distinct points `$$x_k$$`, we can uniquely fit a polynomial of degree `$$ n-1 $$` through those points. The coefficients, e.g., in the monomial basis `$$m_k(x)$$` can be calculated by solving the linear system `$$ [m_k(x_j)] [\hat{m}_k] = [f(x_j)] $$` (i.e. the columns of the matrix are the evaluation of monomial `$$m_k$$` at point `$$x_j$$`). 
An excellent exercise is to formulate this system and solve it in, e.g., python and seeing how it becomes very numerically unstable as `$$n$$` increases, if you choose your interpolation points arbitrarily).

The idea behind Gauss-Lobatto quadrature is to take the roots (zeros) of the `$$m$$`th Legendre polynomial, `$$x_1, \ldots, x_m$$` (when we eventually get to SE, we will include the endpoints of the interval `$$[-1, 1]$$`, but for now we ignore this). Then let `$$q_k$$` be the interpolating polynomial that satisfies `$$$ q_k(x_l) = \begin{cases} 1 \textrm{ if } k=l \\ 0 \textrm{ otherwise} \end{cases}. $$$` 

If we have a polynomial `$$f$$` and its evaluations `$$ f_k = f(x_k)$$`, then if we define `$$w_k = \int_{[-1, 1]} q_k \intd{x} $$` and calculate `$$ \sum_k w_k f_k$$`, then this sum exactly computes `$$ \int_{[-1, 1]} f \intd{x}$$` so long as `$$f$$` is of degree `$$ 2n-1$$` or less. (Note: in practice, `$$w_k$$` can be calculated from the derivatives of legendre polynomials).

Ok, so we have constructed `$$ \sum_k w_k f_k $$` as a way to calculate an integral. It turns out that this also allows us to define an inner product, `$$ \langle f, g \rangle = \sum_k w_k f(x_k) g(x_k) $$`. First observe that `$$\langle q_k, q_l \rangle = \int q_k q_l \intd{x} = \sum_m w_m q_k(x_m) q_l(x_m) = \begin{cases} w_m \textrm{ if } k = l \\ 0 \textrm{ otherwise} \end{cases}  $$`, so `$$q_k$$` are an orthogonal basis.  


Now we're prepared to see why orthonormal bases are so useful. Suppose I have polynomials `$$ h = \sum_k a_k q_k(x) $$`, `$$ g = \sum_k b_k q_k(x)$$`. Then `$$ \langle g,h \rangle = \int \sum_k \sum_{k'} a_k b_{k'} q_k(x) q_{k'}(x) \intd{x} = \sum_k w_k a_k b_k $$`. For a non-orthogonal basis, that inner product operation would require `$$N^2$$` operations, while here it requires only `$$n$$`. 
