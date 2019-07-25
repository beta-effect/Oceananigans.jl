using Printf
using CuArrays, JLD2, FileIO
using Oceananigans

# Physical constants.
ρ₀ = 1027    # Density of seawater [kg/m³]
cₚ = 4181.3  # Specific heat capacity of seawater at constant pressure [J/(kg·K)]
g = 9.8      # Accelration due to gravity
ɑ = 2.1e-4   # Thermal expansion coefficient

# Seconds per day.
spd = 86400
days = 60

Lx, Ly, Lz = 10000, 10000, 2000  # Domain size (meters).
Nx, Ny, Nz = 512, 512, 200  # Number of grid points in each dimension.

# Set up the model and use an artificially high viscosity ν and diffusivity κ.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=GPU(),
              ν=1e-4, κ=1e-4)

# Get location of the cell centers in x, y, z and reshape them to easily
# broadcast over them when calculating hot_bubble_perturbation.
xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
xC, yC, zC = reshape(xC, (Nx, 1, 1)), reshape(yC, (1, Ny, 1)), reshape(zC, (1, 1, Nz))

# Set heating flux at the bottom.
Q = 9200  # W/m^2
Rp = 50   # plume radius [m]

r = @. sqrt((xC - Lx/2)^2 + (yC - Ly/2)^2)

bottom_flux = zeros(Nx, Ny, 1)
bottom_flux[r .< Rp] .= g * ɑ * Q / (ρ₀ * cₚ)
bottom_flux =  CuArray(bottom_flux)

model.boundary_conditions.T.z.right = BoundaryCondition(Flux, bottom_flux)

# Set initial conditions.
N    = 4e-4   # Stratification
dTdz = N^2/g/ɑ
T_prof = 20 .+ dTdz .* model.grid.zC  # Initial temperature profile.
T_3d = repeat(reshape(T_prof, (1, 1, Nz)), Nx, Ny, 1)  # Convert to a 3D array.


# Add small normally distributed random noise to the seafloor to
# facilitate numerical convection.
@. T_3d[:, :, Nz] += 0.001*randn()

@inline ardata_view(f::Field) = view(f.data.parent, 1+f.grid.Hx:f.grid.Nx+f.grid.Hx, 1+f.grid.Hy:f.grid.Ny+f.grid.Hy, 1+f.grid.Hz:f.grid.Nz+f.grid.Hz)
ardata_view(model.tracers.T) .= CuArray(T_3d)

# Add a NaN checker diagnostic that will check for NaN values in the vertical
# velocity and temperature fields every 1,000 time steps and abort the simulation
# if NaN values are detected.
nan_checker = NaNChecker(1000, [model.velocities.w], ["w"])
push!(model.diagnostics, nan_checker)

Δt_wizard = TimeStepWizard(cfl=0.15, Δt=5.0, max_change=1.2, max_Δt=90.0)

# Take Ni "intermediate" time steps at a time before printing a progress
# statement and updating the time step.
Ni = 50

# Write output to disk every No time steps.
No = 1000

end_time = spd * days
while model.clock.time < end_time
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=Δt_wizard.Δt)

    progress = 100 * (model.clock.time / end_time)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt_wizard.Δt / cell_advection_timescale(model)

    update_Δt!(Δt_wizard, model)

    @printf("[%06.2f%%] i: %d, t: %.3f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %6.4g, next Δt: %3.2f s, ⟨wall time⟩: %s",
            progress, model.clock.iteration, model.clock.time / spd,
            umax, vmax, wmax, CFL, Δt_wizard.Δt, prettytime(1e9*walltime / Ni))

    if model.clock.iteration % No == 0
        filename = "hydro_T_plume"  * "_" * string(model.clock.iteration) * ".jld2"
        io_time = @elapsed save(filename,
            Dict("t" => model.clock.time,
                 "xC" => Array(model.grid.xC),
                 "yC" => Array(model.grid.yC),
                 "zC" => Array(model.grid.zC),
                 "xF" => Array(model.grid.xF),
                 "yF" => Array(model.grid.yF),
                 "zF" => Array(model.grid.zF),
                 "u"  => Array(model.velocities.u.data.parent),
                 "v"  => Array(model.velocities.v.data.parent),
                 "w"  => Array(model.velocities.w.data.parent),
                 "T"  => Array(model.tracers.T.data.parent)))
        @printf(", IO time: %s", prettytime(1e9*io_time))
     end
     @printf("\n")
end
