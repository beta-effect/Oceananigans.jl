function test_z_boundary_condition_simple(arch, T, field_name, bctype, bc, Nx, Ny)
    Nz = 16

    bc = BoundaryCondition(bctype, bc)
    modelbcs = BoundaryConditions(field_name, :z, :top, bc)

    model = Model(N=(Nx, Ny, Nz), L=(0.1, 0.2, 0.3), arch=arch, float_type=T,
                  bcs=modelbcs)

    time_step!(model, 1, 1e-16)

    typeof(model) <: Model
end

function test_z_boundary_condition_top_bottom_alias(arch, TF, field_name)
    N = 16
    bcval = 1.0
    top_bc = BoundaryCondition(Value, bcval)
    bottom_bc = BoundaryCondition(Value, -bcval)

    fieldbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
                    top=top_bc, bottom=bottom_bc))

    modelbcs = BoundaryConditions(; Dict((field_name => fieldbcs))...)

    model = Model(N=(N, N, N), L=(0.1, 0.2, 0.3), arch=arch, float_type=TF, bcs=modelbcs)

    bcs = getfield(model.boundary_conditions, field_name)

    time_step!(model, 1, 1e-16)

    getbc(bcs.z.top) == bcval && getbc(bcs.z.bottom) == -bcval
end

function test_z_boundary_condition_array(arch, T, field_name)
    Nx = Ny = Nz = 16

    bcarray = rand(T, Nx, Ny)

    if arch == GPU()
        bcarray = CuArray(bcarray)
    end

    value_bc = BoundaryCondition(Value, bcarray)
    modelbcs = BoundaryConditions(field_name, :z, :top, value_bc)

    model = Model(N=(Nx, Ny, Nz), L=(0.1, 0.2, 0.3), arch=arch, float_type=T, bcs=modelbcs)

    bcs = getfield(model.boundary_conditions, field_name)

    time_step!(model, 1, 1e-16)

    bcs.z.top[1, 2] == bcarray[1, 2]
end

function test_flux_budget(arch, TF, field_name)
    N, κ, Lz = 16, 1, 0.7

    bottom_flux = TF(0.3)
    flux_bc = BoundaryCondition(Flux, bottom_flux)
    modelbcs = BoundaryConditions(field_name, :z, :bottom, flux_bc)

    model = Model(N=(N, N, N), L=(1, 1, Lz), ν=κ, κ=κ,
                  arch=arch, float_type=TF, eos=LinearEquationOfState(βS=0, βT=0),
                  bcs=modelbcs)

    if field_name ∈ (:u, :v, :w)
        field = getfield(model.velocities, field_name)
    else
        field = getfield(model.tracers, field_name)
    end

    @. field.data = 0

    bcs = getfield(model.boundary_conditions, field_name)
    mean_init = mean(data(field))

    τκ = Lz^2 / κ   # Diffusion time-scale
    Δt = 1e-6 * τκ  # Time step much less than diffusion time-scale
    Nt = 100        # Number of time steps

    time_step!(model, Nt, Δt)

    # budget is Lz * ∂<ϕ>/∂t = -Δflux = -top_flux/Lz (left) + bottom_flux/Lz (right);
    # therefore <ϕ> = bottom_flux * t / Lz
    isapprox(mean(data(field)) - mean_init, bottom_flux * model.clock.time / Lz)
end

@testset "Boundary conditions" begin
    println("Testing boundary conditions...")

    funbc(args...) = π

    Nx = Ny = 16
    for arch in archs
        for TF in float_types
            for fld in (:u, :v, :T, :S)
                for bctype in (Gradient, Flux, Value)

                    arraybc = rand(TF, Nx, Ny)
                    if arch == GPU()
                        arraybc = CuArray(arraybc)
                    end

                    for bc in (TF(0.6), arraybc, funbc)
                        @test test_z_boundary_condition_simple(arch, TF, fld, bctype, bc, Nx, Ny)
                    end
                end
                @test test_z_boundary_condition_top_bottom_alias(arch, TF, fld)
                @test test_z_boundary_condition_array(arch, TF, fld)
                @test test_flux_budget(arch, TF, fld)
            end
        end
    end
end
