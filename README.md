# Picograd

An attempt to make a small GPU-accelerated deep-learning framework (mostly) from scratch. The main goals of this project are to have:
- An autograd engine which implements reverse-mode autodiff
- Common operations and network components (`picograd.nn`, `picograd.optim`)
- Rudimentary data processing (i.e. something along the lines of PyTorch's `Dataloader`)
- Avenues such as Numba, CuPy for GPU acceleration