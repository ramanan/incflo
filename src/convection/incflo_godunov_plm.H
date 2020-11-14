#ifndef GODUNOV_PLM_H
#define GODUNOV_PLM_H

#include <AMReX_Gpu.H> 
#include <AMReX_FArrayBox.H>
#include <AMReX_BCRec.H> 
#include <AMReX_BC_TYPES.H>
#include <AMReX_Array.H>
#include <AMReX_Slopes_K.H> 

#ifdef AMREX_USE_EB
#include <AMReX_MultiCutFab.H>
#include <AMReX_EB_slopes_K.H>
#endif

namespace {

// This version is called after the MAC projection
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Godunov_plm_fpu_x (const int i, const int j, const int k, const int n,
                        const amrex::Real dt, const amrex::Real dx,
                        amrex::Real& Im, amrex::Real& Ip,
                        const amrex::Array4<const amrex::Real> &S,
                        const amrex::Real& umac,
                        const amrex::BCRec bc,
                        const int domain_ilo, const int domain_ihi, bool is_velocity = false)
{
    using namespace amrex;
    {
        bool extdir_or_ho_ilo = (bc.lo(0) == BCType::ext_dir) or
                                (bc.lo(0) == BCType::hoextrap);
        bool extdir_or_ho_ihi = (bc.hi(0) == BCType::ext_dir) or
                                (bc.hi(0) == BCType::hoextrap);

        Real upls, umns;
  
        int order = 4; 

        if (i == domain_ilo && (bc.lo(0) == BCType::ext_dir)) {
            umns = S(i-1,j,k,n);
            if(n ==0 and is_velocity)
            {
              upls = S(i-1,j,k,n);
            } else {
            upls = S(i  ,j,k,n) + 0.5 * (-1.0 - umac * dt/dx) *
               amrex_calc_xslope_extdir(i  ,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
            }
        } else if (i == domain_ihi+1 && (bc.hi(0) == BCType::ext_dir)) {
            upls = S(i  ,j,k,n);
            if (n ==0 and is_velocity)
            {
              umns = S(i,j,k,n);
            } else {
               umns = S(i-1,j,k,n) + 0.5 * ( 1.0 - umac * dt/dx) *
               amrex_calc_xslope_extdir(i-1,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
            }
        } else {
            // Note that we still call the "extdir version" here because interior cells one away from the 
            // boundary will still see the boundary condition in the 4th order slope
            upls = S(i  ,j,k,n) + 0.5 * (-1.0 - umac * dt/dx) *
               amrex_calc_xslope_extdir(i  ,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
            umns = S(i-1,j,k,n) + 0.5 * ( 1.0 - umac * dt/dx) *
               amrex_calc_xslope_extdir(i-1,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
        }

        Ip = umns;
        Im = upls;
    }
}

// This version is called after the MAC projection
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Godunov_plm_fpu_y (const int i, const int j, const int k, const int n,
                        const amrex::Real dt, const amrex::Real dy,
                        amrex::Real& Im, amrex::Real& Ip,
                        const amrex::Array4<const amrex::Real> &S,
                        const amrex::Real& vmac,
                        const amrex::BCRec bc,
                        const int domain_jlo, const int domain_jhi, bool is_velocity = false)
{
    using namespace amrex;
    {
        bool extdir_or_ho_jlo = (bc.lo(1) == BCType::ext_dir) or
                                (bc.lo(1) == BCType::hoextrap);
        bool extdir_or_ho_jhi = (bc.hi(1) == BCType::ext_dir) or
                                (bc.hi(1) == BCType::hoextrap);

        Real vpls, vmns;
  
        int order = 4; 

        if (j == domain_jlo && (bc.lo(1) == BCType::ext_dir)) {
            vmns = S(i,j-1,k,n);
            if(n==1 and is_velocity)
            {
                vpls = S(i,j-1,k,n);
            } else {
              vpls = S(i,j  ,k,n) + 0.5 * (-1.0 - vmac * dt/dy) *
                 amrex_calc_yslope_extdir(i,j  ,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            }
        } else if (j == domain_jhi+1 && (bc.hi(1) == BCType::ext_dir)) {
            vpls = S(i,j  ,k,n);
            if(n==1 and is_velocity)
            {
                vmns = S(i,j  ,k,n);
            } else {
            vmns = S(i,j-1,k,n) + 0.5 * ( 1.0 - vmac * dt/dy) *
               amrex_calc_yslope_extdir(i,j-1,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            }
        } else {
            // Note that we still call the "extdir version" here because interior cells one away from the 
            // boundary will still see the boundary condition in the 4th order slope
            vpls = S(i,j  ,k,n) + 0.5 * (-1.0 - vmac * dt/dy) *
               amrex_calc_yslope_extdir(i,j  ,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
            vmns = S(i,j-1,k,n) + 0.5 * ( 1.0 - vmac * dt/dy) *
               amrex_calc_yslope_extdir(i,j-1,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
        }


        Ip = vmns;
        Im = vpls;
    }
}

#ifdef AMREX_USE_EB
//This version compute the slope using least-squares for eb problems
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Godunov_plm_fpu_ls_x (const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Real& umac,
                           const amrex::BCRec bc,
                           amrex::Array4<amrex::EBCellFlag const> const& flag,
                           AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fcx,
                                        amrex::Array4<amrex::Real const> const& fcy,
                                        amrex::Array4<amrex::Real const> const& fcz),
                           amrex::Array4<amrex::Real const> const& ccc,
                           AMREX_D_DECL(const int domain_ilo, 
                                        const int domain_jlo,
                                        const int domain_klo),
                           AMREX_D_DECL(const int domain_ihi, 
                                        const int domain_jhi,
                                        const int domain_khi),
                           bool is_velocity = false)
{
    using namespace amrex;
    {
        bool extdir_or_ho_ilo = (bc.lo(0) == BCType::ext_dir) or
                                (bc.lo(0) == BCType::hoextrap);
        bool extdir_or_ho_ihi = (bc.hi(0) == BCType::ext_dir) or
                                (bc.hi(0) == BCType::hoextrap);

        bool extdir_or_ho_jlo = (bc.lo(1) == BCType::ext_dir) or
                                (bc.lo(1) == BCType::hoextrap);
        bool extdir_or_ho_jhi = (bc.hi(1) == BCType::ext_dir) or
                                (bc.hi(1) == BCType::hoextrap);
        Real upls, umns;
  
        int order = 4; 

        if (i == domain_ilo && (bc.lo(0) == BCType::ext_dir)) {
            umns = S(i-1,j,k,n);
            if(n ==0 and is_velocity)
            {
              upls = S(i-1,j,k,n);
            } else {

              Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to

              const auto& slopes_eb_pls = amrex_calc_slopes_extdir_eb(i,j,k,n,S,ccc,
                                          AMREX_D_DECL(fcx,fcy,fcz), flag,
                                          AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                          AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                          AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                          AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

              AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                           Real yc = ccc(i,j,k,1);,
                           Real zc = ccc(i,j,k,2););

              AMREX_D_TERM(Real delta_x = 0.5 + xc;,
                           Real delta_y = yf  - yc;,
                           Real delta_z = zf  - zc;);

              upls = S(i  ,j,k,n) - delta_x * slopes_eb_pls[0]
                                  + delta_y * slopes_eb_pls[1];

              upls += (-0.5*umac*dt/dx)*slopes_eb_pls[0];

              if (flag(i,j,k).isRegular() and
                  flag(i-1,j,k).isRegular() and
                  flag(i+1,j,k).isRegular() and
                  flag(i-2,j,k).isRegular() and
                  flag(i+2,j,k).isRegular()) { 
                  upls = S(i  ,j,k,n) + 0.5 * (-1.0 - umac * dt/dx) *
                         amrex_calc_xslope_extdir(i  ,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
              }
            }
        } else if (i == domain_ihi+1 && (bc.hi(0) == BCType::ext_dir)) {
            upls = S(i  ,j,k,n);
            if (n ==0 and is_velocity)
            {
              umns = S(i,j,k,n);
            } else {

              Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to

              const auto& slopes_eb_mns =  amrex_calc_slopes_extdir_eb(i-1,j,k,n,S,ccc,
                                           AMREX_D_DECL(fcx,fcy,fcz), flag,
                                           AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                           AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                           AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                           AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

              AMREX_D_TERM(Real xc = ccc(i-1,j,k,0);, // centroid of cell (i,j,k)
                           Real yc = ccc(i-1,j,k,1);,
                           Real zc = ccc(i-1,j,k,2););

              AMREX_D_TERM(Real delta_x = 0.5 - xc;,
                           Real delta_y = yf  - yc;,
                           Real delta_z = zf  - zc;);

              umns = S(i-1,j,k,n) + delta_x * slopes_eb_mns[0]
                                  + delta_y * slopes_eb_mns[1];

              umns += (-0.5*umac*dt/dx)*slopes_eb_mns[0];

              if (flag(i,j,k).isRegular() and
                  flag(i-1,j,k).isRegular() and
                  flag(i+1,j,k).isRegular() and
                  flag(i-2,j,k).isRegular() and
                  flag(i+2,j,k).isRegular()) {
                  umns = S(i-1,j,k,n) + 0.5 * ( 1.0 - umac * dt/dx) *
                  amrex_calc_xslope_extdir(i-1,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
              }
            }
        } else {
            // Note that we still call the "extdir version" here because interior cells one away from the 
            // boundary will still see the boundary condition in the 4th order slope
              Real yf = fcx(i,j,k,0); // local (y,z) of centroid of z-face we are extrapolating to

              const auto& slopes_eb_pls = amrex_calc_slopes_extdir_eb(i,j,k,n,S,ccc,
                                          AMREX_D_DECL(fcx,fcy,fcz), flag,
                                          AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                          AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                          AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                          AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

              AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                           Real yc = ccc(i,j,k,1);,
                           Real zc = ccc(i,j,k,2););

              AMREX_D_TERM(Real delta_x = 0.5 + xc;,
                           Real delta_y = yf  - yc;,
                           Real delta_z = zf  - zc;);

              upls = S(i  ,j,k,n) - delta_x * slopes_eb_pls[0]
                                  + delta_y * slopes_eb_pls[1];

              upls += (-0.5*umac*dt/dx)*slopes_eb_pls[0];

              if (flag(i,j,k).isRegular() and
                  flag(i-1,j,k).isRegular() and
                  flag(i+1,j,k).isRegular() and
                  flag(i-2,j,k).isRegular() and
                  flag(i+2,j,k).isRegular()) {
                  upls = S(i  ,j,k,n) + 0.5 * (-1.0 - umac * dt/dx) *
                     amrex_calc_xslope_extdir(i  ,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
              }

              const auto& slopes_eb_mns =  amrex_calc_slopes_extdir_eb(i-1,j,k,n,S,ccc,
                                           AMREX_D_DECL(fcx,fcy,fcz), flag,
                                           AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                           AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                           AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                           AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

              AMREX_D_TERM(xc = ccc(i-1,j,k,0);, // centroid of cell (i,j,k)
                           yc = ccc(i-1,j,k,1);,
                           zc = ccc(i-1,j,k,2););

              AMREX_D_TERM(delta_x = 0.5 - xc;,
                           delta_y = yf  - yc;,
                           delta_z = zf  - zc;);

              umns = S(i-1,j,k,n) + delta_x * slopes_eb_mns[0]
                                  + delta_y * slopes_eb_mns[1];

              umns += (-0.5*umac*dt/dx)*slopes_eb_mns[0];

              if (flag(i,j,k).isRegular() and
                  flag(i-1,j,k).isRegular() and
                  flag(i+1,j,k).isRegular() and
                  flag(i-2,j,k).isRegular() and
                  flag(i+2,j,k).isRegular()) {
                  umns = S(i-1,j,k,n) + 0.5 * ( 1.0 - umac * dt/dx) *
                       amrex_calc_xslope_extdir(i-1,j,k,n,order,S,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
              }
        }

        Ip = umns;
        Im = upls;
    }
}

//This version compute the slope using least-squares for eb problems
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Godunov_plm_fpu_ls_y (const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dy,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Real& vmac,
                           const amrex::BCRec bc,
                           amrex::Array4<amrex::EBCellFlag const> const& flag,
                           AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fcx, 
                                        amrex::Array4<amrex::Real const> const& fcy, 
                                        amrex::Array4<amrex::Real const> const& fcz),
                           amrex::Array4<amrex::Real const> const& ccc,
                           AMREX_D_DECL(const int domain_ilo, 
                                        const int domain_jlo,
                                        const int domain_klo),
                           AMREX_D_DECL(const int domain_ihi,
                                        const int domain_jhi,
                                        const int domain_khi),
                           bool is_velocity = false)
{
    using namespace amrex;
    {
        bool extdir_or_ho_ilo = (bc.lo(0) == BCType::ext_dir) or
                                (bc.lo(0) == BCType::hoextrap);
        bool extdir_or_ho_ihi = (bc.hi(0) == BCType::ext_dir) or
                                (bc.hi(0) == BCType::hoextrap);

        bool extdir_or_ho_jlo = (bc.lo(1) == BCType::ext_dir) or
                                (bc.lo(1) == BCType::hoextrap);
        bool extdir_or_ho_jhi = (bc.hi(1) == BCType::ext_dir) or
                                (bc.hi(1) == BCType::hoextrap);

        Real vpls, vmns;
  
        int order = 4; 

        if (j == domain_jlo && (bc.lo(1) == BCType::ext_dir)) {
            vmns = S(i,j-1,k,n);
            if(n==1 and is_velocity)
            {
                vpls = S(i,j-1,k,n);
            } else {

                Real xf = fcy(i,j,k,0); // local (x,z) of centroid of z-face we are extrapolating to

                AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                             Real yc = ccc(i,j,k,1);,
                             Real zc = ccc(i,j,k,2););

                AMREX_D_TERM(Real delta_x = xf  - xc;,
                             Real delta_y = 0.5 + yc;,
                             Real delta_z = zf  - zc;);

                // Compute slopes of component "n" of q
                const auto& slopes_eb_pls = amrex_calc_slopes_extdir_eb(i,j,k,n,S,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                vpls = S(i,j  ,k,n) + delta_x * slopes_eb_pls[0]
                                    - delta_y * slopes_eb_pls[1];

                vpls += (-0.5*vmac*dt/dy)*slopes_eb_pls[1];

                if (flag(i,j,k).isRegular() and
                    flag(i,j-1,k).isRegular() and
                    flag(i,j+1,k).isRegular() and
                    flag(i,j-2,k).isRegular() and
                    flag(i,j+2,k).isRegular()) {
                    vpls = S(i,j  ,k,n) + 0.5 * (-1.0 - vmac * dt/dy) *
                       amrex_calc_yslope_extdir(i,j  ,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
                }
            }
        } else if (j == domain_jhi+1 && (bc.hi(1) == BCType::ext_dir)) {
            vpls = S(i,j  ,k,n);
            if(n==1 and is_velocity)
            {
                vmns = S(i,j  ,k,n);
            } else {

                Real xf = fcy(i,j,k,0); // local (x,z) of centroid of z-face we are extrapolating to

                AMREX_D_TERM(Real xc = ccc(i,j-1,k,0);, // centroid of cell (i,j,k)
                             Real yc = ccc(i,j-1,k,1);,
                             Real zc = ccc(i,j-1,k,2););

                AMREX_D_TERM(Real delta_x = xf  - xc;,
                             Real delta_y = 0.5 - yc;,
                             Real delta_z = zf  - zc;);

                // Compute slopes of component "n" of q
                const auto& slopes_eb_mns = amrex_calc_slopes_extdir_eb(i,j-1,k,n,S,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                vmns = S(i,j-1,k,n) + delta_x * slopes_eb_mns[0]
                                    + delta_y * slopes_eb_mns[1];

                vmns += (-0.5*vmac*dt/dy)*slopes_eb_mns[1];

                if (flag(i,j,k).isRegular() and
                    flag(i,j-1,k).isRegular() and
                    flag(i,j+1,k).isRegular() and
                    flag(i,j-2,k).isRegular() and
                    flag(i,j+2,k).isRegular()) {
                    vmns = S(i,j-1,k,n) + 0.5 * ( 1.0 - vmac * dt/dy) *
                      amrex_calc_yslope_extdir(i,j-1,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
                }
            }
        } else {
            // Note that we still call the "extdir version" here because interior cells one away from the 
            // boundary will still see the boundary condition in the 4th order slope
                Real xf = fcy(i,j,k,0); // local (x,z) of centroid of z-face we are extrapolating to

                AMREX_D_TERM(Real xc = ccc(i,j,k,0);, // centroid of cell (i,j,k)
                             Real yc = ccc(i,j,k,1);,
                             Real zc = ccc(i,j,k,2););

                AMREX_D_TERM(Real delta_x = xf  - xc;,
                             Real delta_y = 0.5 + yc;,
                             Real delta_z = zf  - zc;);

                // Compute slopes of component "n" of q
                const auto& slopes_eb_pls = amrex_calc_slopes_extdir_eb(i,j,k,n,S,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                vpls = S(i,j  ,k,n) + delta_x * slopes_eb_pls[0]
                                    - delta_y * slopes_eb_pls[1];

                vpls += (-0.5*vmac*dt/dy)*slopes_eb_pls[1];

                if (flag(i,j,k).isRegular() and
                    flag(i,j-1,k).isRegular() and
                    flag(i,j+1,k).isRegular() and
                    flag(i,j-2,k).isRegular() and
                    flag(i,j+2,k).isRegular()) {
                    vpls = S(i,j  ,k,n) + 0.5 * (-1.0 - vmac * dt/dy) *
                           amrex_calc_yslope_extdir(i,j  ,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
                }

                AMREX_D_TERM(xc = ccc(i,j-1,k,0);, // centroid of cell (i,j,k)
                             yc = ccc(i,j-1,k,1);,
                             zc = ccc(i,j-1,k,2););

                AMREX_D_TERM(delta_x = xf  - xc;,
                             delta_y = 0.5 - yc;,
                             delta_z = zf  - zc;);

                // Compute slopes of component "n" of q
                const auto& slopes_eb_mns = amrex_calc_slopes_extdir_eb(i,j-1,k,n,S,ccc,
                                            AMREX_D_DECL(fcx,fcy,fcz), flag,
                                            AMREX_D_DECL(extdir_or_ho_ilo, extdir_or_ho_jlo, extdir_or_ho_klo),
                                            AMREX_D_DECL(extdir_or_ho_ihi, extdir_or_ho_jhi, extdir_or_ho_khi),
                                            AMREX_D_DECL(domain_ilo, domain_jlo, domain_klo),
                                            AMREX_D_DECL(domain_ihi, domain_jhi, domain_khi));

                vmns = S(i,j-1,k,n) + delta_x * slopes_eb_mns[0]
                                    + delta_y * slopes_eb_mns[1];

                vmns += (-0.5*vmac*dt/dy)*slopes_eb_mns[1];

                if (flag(i,j,k).isRegular() and
                    flag(i,j-1,k).isRegular() and
                    flag(i,j+1,k).isRegular() and
                    flag(i,j-2,k).isRegular() and
                    flag(i,j+2,k).isRegular()) {
                    vmns = S(i,j-1,k,n) + 0.5 * ( 1.0 - vmac * dt/dy) *
                         amrex_calc_yslope_extdir(i,j-1,k,n,order,S,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
                }
        }


        Ip = vmns;
        Im = vpls;
    }
}
#endif

#if (AMREX_SPACEDIM == 3)
// This version is called after the MAC projection
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Godunov_plm_fpu_z (const int i, const int j, const int k, const int n,
                        const amrex::Real dt, const amrex::Real dz,
                        amrex::Real& Im, amrex::Real& Ip,
                        const amrex::Array4<const amrex::Real> &S,
                        const amrex::Real& wmac,
                        const amrex::BCRec bc,
                        const int domain_klo, const int domain_khi, bool is_velocity = false)
{
    using namespace amrex;
    {
        bool extdir_or_ho_klo = (bc.lo(2) == BCType::ext_dir) or
                                (bc.lo(2) == BCType::hoextrap);
        bool extdir_or_ho_khi = (bc.hi(2) == BCType::ext_dir) or
                                (bc.hi(2) == BCType::hoextrap);

        Real wpls, wmns;
  
        int order = 4; 

        if (k == domain_klo && (bc.lo(2) == BCType::ext_dir)) {
            wmns = S(i,j,k-1,n);
            if(n==2 and is_velocity)
            {
               wpls = S(i,j,k-1,n);
            } else {
               wpls = S(i,j,k  ,n) + 0.5 * (-1.0 - wmac * dt/dz) *
                  amrex_calc_zslope_extdir(i,j,k  ,n,order,S,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            }
        } else if (k == domain_khi+1 && (bc.hi(2) == BCType::ext_dir)) {
            wpls = S(i,j,k  ,n);
            if(n==2 and is_velocity)
            {
                wmns = S(i,j,k  ,n);
            } else {
               wmns = S(i,j,k-1,n) + 0.5 * ( 1.0 - wmac * dt/dz) *
                 amrex_calc_zslope_extdir(i,j,k-1,n,order,S,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            }
        } else {
            // Note that we still call the "extdir version" here because interior cells one away from the 
            // boundary will still see the boundary condition in the 4th order slope
            wpls = S(i,j,k  ,n) + 0.5 * (-1.0 - wmac * dt/dz) *
               amrex_calc_zslope_extdir(i,j,k  ,n,order,S,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
            wmns = S(i,j,k-1,n) + 0.5 * ( 1.0 - wmac * dt/dz) *
               amrex_calc_zslope_extdir(i,j,k-1,n,order,S,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
        }


        Ip = wmns;
        Im = wpls;
    }
}
#endif

}
#endif 
