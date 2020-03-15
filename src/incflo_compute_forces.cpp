#include <incflo.H>

using namespace amrex;

void incflo::compute_tra_forces (Vector<MultiFab*> const& tra_forces)
{
    // For now we don't have any external forces on the scalars
    if (m_advect_tracer) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            tra_forces[lev]->setVal(0.0);
        }
    }
}

void incflo::compute_vel_forces (Vector<MultiFab*> const& vel_forces,
                                 Vector<MultiFab const*> const& velocity,
                                 Vector<MultiFab const*> const& density,
                                 Vector<MultiFab const*> const& tracer_old,
                                 Vector<MultiFab const*> const& tracer_new)
{
    for (int lev = 0; lev <= finest_level; ++lev)  
       compute_vel_forces_on_level (lev, *vel_forces[lev], *velocity[lev], *density[lev], 
                                         *tracer_old[lev], *tracer_new[lev]);
}

void incflo::compute_vel_forces_on_level (int lev,
                                                MultiFab& vel_forces,
                                          const MultiFab& velocity,
                                          const MultiFab& density,
                                          const MultiFab& tracer_old,
                                          const MultiFab& tracer_new)
{
    const Real* dx = geom[lev].CellSize();

    GpuArray<Real,3> l_gravity{m_gravity[0],m_gravity[1],m_gravity[2]};
    GpuArray<Real,3> l_gp0{m_gp0[0], m_gp0[1], m_gp0[2]};

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(vel_forces,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
            Box const& bx = mfi.tilebox();
            Array4<Real>       const& vel_f =  vel_forces.array(mfi);
            Array4<Real const> const&   vel =    velocity.const_array(mfi);
            Array4<Real const> const&   rho =     density.const_array(mfi);
            Array4<Real const> const& gradp = m_leveldata[lev]->gp.const_array(mfi);

            if (m_use_boussinesq) {
                // This uses a Boussinesq approximation where the buoyancy depends on
                //      first tracer rather than density
                Array4<Real const> const& tra_o = tracer_old.const_array(mfi);
                Array4<Real const> const& tra_n = tracer_new.const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real rhoinv = 1.0/rho(i,j,k);
                    Real ft = 0.5 * (tra_o(i,j,k) + tra_n(i,j,k));

                    vel_f(i,j,k,0) = -gradp(i,j,k,0)*rhoinv + l_gravity[0] * ft;
                    vel_f(i,j,k,1) = -gradp(i,j,k,1)*rhoinv + l_gravity[1] * ft;
                    vel_f(i,j,k,2) = -gradp(i,j,k,2)*rhoinv + l_gravity[2] * ft; 
                });

            } else {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real rhoinv = 1.0/rho(i,j,k);

                    vel_f(i,j,k,0) = -(gradp(i,j,k,0)+l_gp0[0])*rhoinv + l_gravity[0];
                    vel_f(i,j,k,1) = -(gradp(i,j,k,1)+l_gp0[1])*rhoinv + l_gravity[1];
                    vel_f(i,j,k,2) = -(gradp(i,j,k,2)+l_gp0[2])*rhoinv + l_gravity[2];
                });
            }
    }
}
