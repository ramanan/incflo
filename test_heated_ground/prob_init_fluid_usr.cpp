#include <incflo.H>

using namespace amrex;

void incflo::init_heated_ground (Box const& vbx, Box const& gbx,
                                 Array4<Real> const& p,
                                 Array4<Real> const& vel,
                                 Array4<Real> const& density,
                                 Array4<Real> const& tracer,
                                 Box const& domain,
                                 GpuArray<Real, AMREX_SPACEDIM> const& dx,
                                 GpuArray<Real, AMREX_SPACEDIM> const& problo,
                                 GpuArray<Real, AMREX_SPACEDIM> const& probhi)
{
     AMREX_D_TERM(Real u = m_ic_u;,
                  Real v = m_ic_v;,
                  Real w = m_ic_w;);
     Real init_temp = m_ic_t;
     amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        AMREX_D_TERM(vel(i,j,k,0) = u;,
                     vel(i,j,k,1) = v;,
                     vel(i,j,k,2) = w;);
        tracer(i,j,k) = init_temp;
     });
}
