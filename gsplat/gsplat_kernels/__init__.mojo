"""GPU kernels for 3D gaussian splatting.

Packaged so the modules can be precompiled into a `.mojoc` and loaded as a
MAX custom extension; MAX needs a real package, not a directory of scripts.
Executables with a `main()` live in `tests/` instead, since a package cannot
contain one.
"""
