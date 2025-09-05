# deep-bsde-basket

Implemented and extended a Deep BSDE solver (E, Han & Jentzen, 2017–2018; extended from https://github.com/YifanJiang233/Deep_BSDE_solver — a
neural network method for solving pricing PDEs via BSDEs — for multi-asset European basket calls in PyTorch. Added
exact GBM stepping, correlated stocks, and delta extraction; benchmarked against Monte Carlo and Black
Scholes.
