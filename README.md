# pemfc-dedalus

3D PEMFC multiphysics modelling with Dedalus.

## Target cell

The target architecture is a **forced-air open-cathode, dead-end-anode PEMFC**:

- cathode air provides both oxygen supply and cell cooling;
- the hydrogen anode has no continuous outlet flow during normal operation;
- anode water / inert accumulation will later be handled with explicit purge events;
- the present models are built incrementally so each physical block can be validated.

## Implemented models

### V0.1 — cathode O2 diffusion/reaction validation

`pemfc-cathode-3d` solves 3D oxygen diffusion through cathode GDL + CL with a simple
first-order reaction sink.

### V0.2 — cathode electrochemistry

`pemfc-cathode-electrochem-3d` adds:

- O2 transport in the GDL and catalyst layer;
- electronic potential `phi_s`;
- protonic potential `phi_m`;
- concentration-dependent Butler-Volmer oxygen-reduction kinetics;
- volumetric ORR current density `j_orr(x,y,z)`;
- HDF5 outputs for `c_o2`, `phi_s`, `phi_m`, overpotential and reaction current.

The GDL/CL transition is currently regularised smoothly in one Chebyshev domain.
Proton conductivity outside the CL uses a small numerical floor to avoid a singular
global operator. This is a deliberate validation approximation and will be removed
when the MEA is split into proper spectral subdomains.

The electric equations use pseudo-transient continuation. Their converged state is
the stationary conduction solution; this pseudo-time is **not** yet a physical
double-layer transient.

## Installation on Debian

Dedalus requires MPI and FFTW, including FFTW's MPI headers:

```bash
sudo apt install build-essential python3-dev python3-venv \
  libopenmpi-dev openmpi-bin libfftw3-dev libfftw3-mpi-dev
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

export CC=mpicc
export MPICC=mpicc
export CFLAGS="$(mpicc --showme:compile)"
export LDFLAGS="$(mpicc --showme:link)"

python -m pip install -e .
```

For Dedalus runs:

```bash
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Run V0.1

```bash
pemfc-cathode-3d --nx 16 --ny 16 --nz 48 --stop-time 0.02
```

## Run V0.2

Start small:

```bash
pemfc-cathode-electrochem-3d --nx 16 --ny 16 --nz 48 \
  --stop-time 0.002 --max-dt 2e-6
```

On the 4-core / 8-thread test host, use all logical CPUs when wall-clock time matters:

```bash
mpiexec --use-hwthread-cpus -n 8 \
  pemfc-cathode-electrochem-3d --nx 64 --ny 64 --nz 96 \
  --stop-time 0.002 --max-dt 2e-6
```

## Coordinates

- `x`: longitudinal direction;
- `y`: transverse direction over the open-cathode face;
- `z`: through-plane direction;
- `z=0`: cathode air/GDL side;
- `z=Lz`: CL/membrane side.

The smooth x/y oxygen modulation in the current validation problem represents
nonuniform open-cathode airflow/exposure. A real forced-air velocity and temperature
field will replace it when the channel/free-air momentum and thermal model is added.

## V0.2 equations

Oxygen:

```text
eps * dc_O2/dt = div(D_eff grad(c_O2)) - j_ORR/(4F)
```

Solid conduction at steady state:

```text
div(sigma_s grad(phi_s)) = j_ORR
```

Protonic conduction at steady state:

```text
div(sigma_m grad(phi_m)) = -j_ORR
```

Cathodic Butler-Volmer magnitude:

```text
eta = phi_s - phi_m - E_eq

j_ORR = chi_CL * j0_vol * (c_O2/c_ref)^gamma *
        [exp(-alpha_c F eta / RT) - exp(alpha_a F eta / RT)]
```

## Roadmap

1. Validate V0.2 current and potential profiles and add post-processing.
2. Split GDL / CL / PEM into proper spectral subdomains.
3. Add membrane water content `lambda`, back diffusion and electro-osmotic drag.
4. Add dead-end anode H2 transport + HOR.
5. Add discrete/controlled purge events for anode H2O/inert removal.
6. Add forced-air cathode momentum transport.
7. Add the thermal equation; cathode air is both oxidant and coolant.
8. Couple air-flow rate to O2 availability and convective heat removal.
9. Add liquid-water saturation / capillary transport.
10. Calibrate against polarization and transient experimental data.

## License

No license has been selected yet.
