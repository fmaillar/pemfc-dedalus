# pemfc-dedalus

3D PEMFC multiphysics modelling experiments with [Dedalus](https://dedalus-project.org/).

## Status

This repository is intentionally built in verified increments.

**V0.1** implements a first 3D cathode validation problem:

- periodic spectral directions in `x` and `y`;
- Chebyshev collocation through the thickness `z`;
- oxygen diffusion through a porous GDL + catalyst layer;
- smooth GDL/CL material transition;
- effective diffusivity using a Bruggeman correction;
- oxygen consumption localised in the catalyst layer;
- channel/rib modulation at the gas-side boundary;
- time integration to approach the stationary diffusion-reaction state;
- HDF5 snapshots and scalar diagnostics.

This is **not yet the complete PEMFC model**. The next increments will add solid/protonic
potentials and Butler-Volmer kinetics, membrane water transport (back diffusion +
electro-osmotic drag), the anode, thermal transport, gas flow, and finally liquid-water
transport.

## Governing equation in V0.1

The solved field is the oxygen molar concentration
\(c_{O_2}(x,y,z,t)\):

\[
\varepsilon \frac{\partial c_{O_2}}{\partial t}
=
\nabla \cdot (D_{\mathrm{eff}}\nabla c_{O_2})
-k_{\mathrm{rxn}}\chi_{\mathrm{CL}}c_{O_2}.
\]

For this first validation case, the GDL/CL interface is regularised by a smooth
indicator \(\chi_{\mathrm{CL}}(z)\). This avoids Gibbs oscillations from a discontinuous
material coefficient in a single Chebyshev domain.

The effective diffusivity is

\[
D_{\mathrm{eff}} = D_{O_2}\varepsilon^{3/2}.
\]

The cathode gas boundary imposes an O2 concentration modulated in \(y\), so that the
3D solver sees a channel/rib-like nonuniformity. The CL/membrane side uses zero O2 flux.

## Installation

Dedalus depends on MPI and FFTW. On Debian, install the native dependencies first:

```bash
sudo apt install build-essential python3-dev python3-venv libopenmpi-dev openmpi-bin libfftw3-dev
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Check the import:

```bash
python -c 'import dedalus.public as d3; print("Dedalus OK")'
```

## Run

A small serial smoke test:

```bash
pemfc-cathode-3d --nx 16 --ny 16 --nz 48 --stop-time 0.02
```

or directly:

```bash
python -m pemfc_dedalus.cathode_3d --nx 16 --ny 16 --nz 48 --stop-time 0.02
```

For MPI:

```bash
mpiexec -n 4 pemfc-cathode-3d --nx 64 --ny 64 --nz 96 --stop-time 0.2
```

Results are written below `output/`.

## Numerical coordinates

- `x`: streamwise direction;
- `y`: transverse channel/rib direction;
- `z`: through-plane direction, gas side at `z = 0`, membrane side at `z = Lz`.

The present geometry is a representative periodic cathode patch, not a full serpentine
flow field.

## Development roadmap

1. Validate 3D O2 diffusion-reaction against a 1D analytical limit.
2. Add electronic potential \(\phi_s\).
3. Add protonic potential \(\phi_m\).
4. Replace the linear reaction by concentration-dependent Butler-Volmer kinetics.
5. Split/verify GDL and CL interface treatment.
6. Add PEM water content \(\lambda\), back diffusion and electro-osmotic drag.
7. Add anode H2 transport and HOR.
8. Add heat equation and temperature-dependent material properties.
9. Add channel/GDL momentum transport (Navier-Stokes + Brinkman/Darcy).
10. Add liquid-water saturation and capillary transport.

## License

No license has been selected yet.
