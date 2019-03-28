C Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
C the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
C reserved. See files LICENSE and NOTICE for details.
C
C This file is part of CEED, a collection of benchmarks, miniapps, software
C libraries and APIs for efficient high-order finite element and spectral
C element discretizations for exascale applications. For more information and
C source code availability see http://github.com/ceed.
C
C The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
C a collaborative effort of two U.S. Department of Energy organizations (Office
C of Science and the National Nuclear Security Administration) responsible for
C the planning and preparation of a capable exascale ecosystem, including
C software, applications, hardware, advanced system engineering and early
C testbed platforms, in support of the nation's exascale computing imperative.

C> @file
C> Mass operator example using Nek5000
C TESTARGS -c {ceed_resource} -e bp1 -n 1 -b 6 -test

C-----------------------------------------------------------------------
      subroutine masssetupf(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
C     Set up mass operator
C     Input: u1,u2,u3,q             Output: v1,v2,ierr
      integer q,ierr
      real*8 ctx(1)
      real*8 u1(3*q)
      real*8 u2(9*q)
      real*8 u3(q)
      real*8 v1(q)
      real*8 v2(q)
      real*8 a11,a12,a13,a21,a22,a23,a31,a32,a33
      real*8 g11,g12,g13,g21,g22,g23,g31,g32,g33
      real*8 jacmq

      do i=1,q
        a11=u2(i+q*0)
        a12=u2(i+q*3)
        a13=u2(i+q*6)

        a21=u2(i+q*1)
        a22=u2(i+q*4)
        a23=u2(i+q*7)

        a31=u2(i+q*2)
        a32=u2(i+q*5)
        a33=u2(i+q*8)

        g11 = (a22*a33-a23*a32)
        g12 = (a13*a32-a33*a12)
        g13 = (a12*a23-a22*a13)

        g21 = (a23*a31-a21*a33)
        g22 = (a11*a33-a31*a13)
        g23 = (a13*a21-a23*a11)

        g31 = (a21*a32-a22*a31)
        g32 = (a12*a31-a32*a11)
        g33 = (a11*a22-a21*a12)

        jacmq = a11*g11+a21*g12+a31*g13

C       Rho
        v1(i)=u3(i)*jacmq

C       RHS
        v2(i)=u3(i)*jacmq
     $             *dsqrt(u1(i+q*0)*u1(i+q*0)
     $                   +u1(i+q*1)*u1(i+q*1)
     $                   +u1(i+q*2)*u1(i+q*2))
      enddo

      ierr=0
      end
C-----------------------------------------------------------------------
      subroutine massf(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
C     Apply mass operator
C     Input: u1,u2,q                Output: v1,ierr
      integer q,ierr
      real*8 ctx(1)
      real*8 u1(q)
      real*8 u2(q)
      real*8 v1(q)

      do i=1,q
        v1(i)=u2(i)*u1(i)
      enddo

      ierr=0
      end
C-----------------------------------------------------------------------
      subroutine diffsetupf(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
C     Set up diffusion operator
C     Input: u1,u2,u3,q             Output: v1,v2,ierr
      integer q,ierr
      real*8 ctx(1)
      real*8 u1(3*q)
      real*8 u2(9*q)
      real*8 u3(q)
      real*8 v1(6*q)
      real*8 v2(q)
      real*8 a11,a12,a13,a21,a22,a23,a31,a32,a33
      real*8 g11,g12,g13,g21,g22,g23,g31,g32,g33
      real*8 jacmq,scl
      real*8 c(3),k(3)

      do i=1,q
        pi = 3.14159265358979323846

        c(1)=0.
        c(2)=1.
        c(3)=2.
        k(1)=1.
        k(2)=2.
        k(3)=3.

        a11=u2(i+q*0)
        a12=u2(i+q*3)
        a13=u2(i+q*6)

        a21=u2(i+q*1)
        a22=u2(i+q*4)
        a23=u2(i+q*7)

        a31=u2(i+q*2)
        a32=u2(i+q*5)
        a33=u2(i+q*8)

        g11 = (a22*a33-a23*a32)
        g12 = (a13*a32-a33*a12)
        g13 = (a12*a23-a22*a13)

        g21 = (a23*a31-a21*a33)
        g22 = (a11*a33-a31*a13)
        g23 = (a13*a21-a23*a11)

        g31 = (a21*a32-a22*a31)
        g32 = (a12*a31-a32*a11)
        g33 = (a11*a22-a21*a12)

        jacmq = a11*g11+a21*g12+a31*g13

        scl = u3(i)/jacmq

C       Geometric factors
        v1(i+0*q) = scl*(g11*g11+g12*g12+g13*g13) ! Grr
        v1(i+1*q) = scl*(g11*g21+g12*g22+g13*g23) ! Grs
        v1(i+2*q) = scl*(g11*g31+g12*g32+g13*g33) ! Grt
        v1(i+3*q) = scl*(g21*g21+g22*g22+g23*g23) ! Gss
        v1(i+4*q) = scl*(g21*g31+g22*g32+g23*g33) ! Gst
        v1(i+5*q) = scl*(g31*g31+g32*g32+g33*g33) ! Gtt

C       RHS
        v2(i) = u3(i)*jacmq*pi*pi
     $            *dsin(pi*(c(1)+k(1)*u1(i+0*q)))
     $            *dsin(pi*(c(2)+k(2)*u1(i+1*q)))
     $            *dsin(pi*(c(3)+k(3)*u1(i+2*q)))  
     $            *(k(1)*k(1)+k(2)*k(2)+k(3)*k(3)) 

      enddo

      ierr=0
      end
C-----------------------------------------------------------------------
      subroutine diffusionf(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
C     Apply diffusion operator
C     Input: u1,u2,q                Output: v1,ierr
      integer q,ierr
      real*8 ctx(1)
      real*8 u1(3*q)
      real*8 u2(6*q)
      real*8 v1(3*q)

      do i=1,q
        v1(i+0*q)=
     $     u2(i+0*q)*u1(i)+u2(i+1*q)*u1(i+q)+u2(i+2*q)*u1(i+2*q)
        v1(i+1*q)=
     $     u2(i+1*q)*u1(i)+u2(i+3*q)*u1(i+q)+u2(i+4*q)*u1(i+2*q)
        v1(i+2*q)=
     $     u2(i+2*q)*u1(i)+u2(i+4*q)*u1(i+q)+u2(i+5*q)*u1(i+2*q)
      enddo

      ierr=0
      end
C-----------------------------------------------------------------------
      subroutine set_h2_as_rhoJac_GL(h2,bmq,nxq)
C     Set h2 as rhoJac
C     Input: bmq,nxq                Output: h2
      include 'SIZE'
      real*8 h2(1),bmq(1)

      common /ctmp77/ wd(lxd),zd(lxd)
      integer e,i,L

      call zwgl(zd,wd,nxq)  ! nxq = number of points

      q = 1.0               ! Later, this can be a function of position...

      L = 0
      do e=1,lelt
      do i=1,nxq**ldim
         L=L+1
         h2(L) = q*bmq(L)
      enddo
      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine dist_fld_h1(e)
C     Set distance initial condition for BP1
C     Input:                        Output: e
      include 'SIZE'
      include 'TOTAL'
      real*8 x,y,z
      real*8 e(1)

      n=lx1*ly1*lz1*nelt

      do i=1,n
        x=xm1(i,1,1,1)
        y=ym1(i,1,1,1)
        z=zm1(i,1,1,1)

        e(i) = dsqrt(x*x+y*y+z*z)
      enddo

      call dsavg(e)  ! This is requisite for random fields

      return
      end
C-----------------------------------------------------------------------
      subroutine sin_fld_h1(e)
C     Set sine initial condition for BP3
C     Input:                        Output: e
      include 'SIZE'
      include 'TOTAL'
      real*8 x,y,z
      real*8 e(1)
      real*8 c(3),k(3)

      n=lx1*ly1*lz1*nelt
      pi = 3.14159265358979323846

      c(1)=0.
      c(2)=1.
      c(3)=2.
      k(1)=1.
      k(2)=2.
      k(3)=3.

      do i=1,n
        x=xm1(i,1,1,1)
        y=ym1(i,1,1,1)
        z=zm1(i,1,1,1)

        e(i) = dsin(pi*(c(1)+k(1)*x))
     &        *dsin(pi*(c(2)+k(2)*y))
     &        *dsin(pi*(c(3)+k(3)*z))

      enddo

      call dsavg(e)  ! This is requisite for random fields

      return
      end
C-----------------------------------------------------------------------
      subroutine uservp(ix,iy,iz,eg) ! set variable properties
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,f,eg
C     e = gllel(eg)

      udiff  = 0.0
      utrans = 0.0

      return
      end
C-----------------------------------------------------------------------
      subroutine userf(ix,iy,iz,eg) ! set acceleration term
C
C     Note: this is an acceleration term, NOT a force!
C     Thus, ffx will subsequently be multiplied by rho(x,t).
C
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,f,eg
C     e = gllel(eg)

      ffx = 0.0
      ffy = 0.0
      ffz = 0.0

      return
      end
C-----------------------------------------------------------------------
      subroutine userq(i,j,k,eg) ! set source term
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,f,eg
      e = gllel(eg)

      qvol   = 0

      return
      end
C-----------------------------------------------------------------------
      subroutine userbc(ix,iy,iz,f,eg) ! set up boundary conditions
C     NOTE ::: This subroutine MAY NOT be called by every process
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,f,eg

      ux   = 0.0
      uy   = 0.0
      uz   = 0.0
      temp = 0.0

      return
      end
C-----------------------------------------------------------------------
      subroutine useric(ix,iy,iz,eg) ! set up initial conditions
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,f,eg

      ux   = 0.0
      uy   = 0.0
      uz   = 0.0
      temp = 0.0

      return
      end
C-----------------------------------------------------------------------
      subroutine usrdat   ! This routine to modify element vertices
      include 'SIZE'
      include 'TOTAL'

      return
      end
C-----------------------------------------------------------------------
      subroutine usrdat2  ! This routine to modify mesh coordinates
      include 'SIZE'
      include 'TOTAL'

      x0 = 0
      x1 = 1
      call rescale_x(xm1,x0,x1)
      call rescale_x(ym1,x0,x1)
      call rescale_x(zm1,x0,x1)

      param(59)=1  ! Force Nek to use the "deformed element" formulation

      return
      end
C-----------------------------------------------------------------------
      subroutine usrdat3
      include 'SIZE'
      include 'TOTAL'

      return
      end
C-----------------------------------------------------------------------
      subroutine xmask1   (r1,h2,nel)
C     Apply zero Dirichlet boundary conditions
C     Input: h2,nel                 Output: r1
      include 'SIZE'
      include 'TOTAL'
      real*8 r1(1),h2(1)

      n=nx1*ny1*nz1*nel
      do i=1,n
         r1(i)=r1(i)*h2(i)
      enddo

      return
      end
C-----------------------------------------------------------------------
      function glrdif(x,y,n)
C     Compute Linfty norm of (x-y)
C     Input: x,y                    Output: n
      real*8 x(n),y(n)

      dmx=0
      xmx=0
      ymx=0

      do i=1,n
         diff=abs(x(i)-y(i))
         dmx =max(dmx,diff)
         xmx =max(xmx,x(i))
         ymx =max(ymx,y(i))
      enddo

      xmx = max(xmx,ymx)
      dmx = glmax(dmx,1) ! max across processors
      xmx = glmax(xmx,1)

      if (xmx.gt.0) then
         glrdif = dmx/xmx
      else
         glrdif = -dmx   ! Negative indicates something strange
      endif

      return
      end
C-----------------------------------------------------------------------
      subroutine loc_grad3(ur,us,ut,u,N,D,Dt)
C     3D transpose of local gradient
C     Input: u,N,D,Dt               Output: ur,us,ut
      real*8 ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real*8 u (0:N,0:N,0:N)
      real*8 D (0:N,0:N),Dt(0:N,0:N)

      m1 = N+1
      m2 = m1*m1

      call mxm(D ,m1,u(0,0,0),m1,ur,m2)
      do k=0,N
         call mxm(u(0,0,k),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call mxm(u(0,0,0),m2,Dt,m1,ut,m1)

      return
      end
c-----------------------------------------------------------------------
      subroutine loc_grad3t(u,ur,us,ut,N,D,Dt,w)
C     3D transpose of local gradient
C     Input: ur,us,ut,N,D,Dt        Output: u
       real*8 u (0:N,0:N,0:N)
       real*8 ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
       real*8 D (0:N,0:N),Dt(0:N,0:N)
       real*8 w (0:N,0:N,0:N)
 
       m1 = N+1
       m2 = m1*m1
       m3 = m1*m1*m1
 
       call mxm(Dt,m1,ur,m1,u(0,0,0),m2)
       do k=0,N
          call mxm(us(0,0,k),m1,D ,m1,w(0,0,k),m1)
       enddo
       call add2(u(0,0,0),w,m3)
       call mxm(ut,m2,D ,m1,w,m1)
       call add2(u(0,0,0),w,m3)

      return
      end
C-----------------------------------------------------------------------
      subroutine geodatq(gf,bmq,w3mq,nxq)
C     Routine to generate elemental geometric matrices on mesh 1
C     (Gauss-Legendre Lobatto mesh).
      include 'SIZE'
      include 'TOTAL'

      parameter (lg=3+3*(ldim-2),lzq=lx1+1,lxyd=lzq**ldim)

      real*8 gf(lg,nxq**ldim,lelt),bmq(nxq**ldim,lelt),w3mq(nxq,nxq,nxq)

      common /ctmp1/ xr(lxyd),xs(lxyd),xt(lxyd)
      common /sxrns/ yr(lxyd),ys(lxyd),yt(lxyd)
     $ ,             zr(lxyd),zs(lxyd),zt(lxyd)

      common /ctmp77/ wd(lxd),zd(lxd)
      common /dxmfine/ dxmq(lzq,lzq),dxtmq(lzq,lzq)

      integer e
      real*8 tmp(lxyd)
      real*8 a11,a12,a13,a21,a22,a23,a31,a32,a33
      real*8 g11,g12,g13,g21,g22,g23,g31,g32,g33
      real*8 jacmq

      if (nxq.gt.lzq) call exitti('ABORT: recompile with lzq=$',nxq)

      call zwgl    (zd,wd,lzq)                            ! nxq = number of points
      call gen_dgl (dxmq,dxtmq,lzq,lzq,tmp)

      do k=1,nxq
      do j=1,nxq
      do i=1,nxq
         w3mq(i,j,k) = wd(i)*wd(j)*wd(k)
      enddo
      enddo
      enddo

      nxyzq = nxq**ldim
      nxqm1 = lzq-1

      do e=1,nelt
         call intp_rstd (tmp,xm1(1,1,1,e),lx1,lzq,if3d,0) ! 0-->Fwd interpolation
         call loc_grad3 (xr,xs,xt,tmp,nxqm1,dxmq,dxtmq)

         call intp_rstd (tmp,ym1(1,1,1,e),lx1,lzq,if3d,0)
         call loc_grad3 (yr,ys,yt,tmp,nxqm1,dxmq,dxtmq)

         call intp_rstd (tmp,zm1(1,1,1,e),lx1,lzq,if3d,0)
         call loc_grad3 (zr,zs,zt,tmp,nxqm1,dxmq,dxtmq)

         do i=1,nxyzq
            a11 = xr(i)
            a12 = xs(i)
            a13 = xt(i)

            a21 = yr(i)
            a22 = ys(i)
            a23 = yt(i)

            a31 = zr(i)
            a32 = zs(i)
            a33 = zt(i)

            g11 = (a22*a33-a23*a32)
            g12 = (a13*a32-a33*a12)
            g13 = (a12*a23-a22*a13)

            g21 = (a23*a31-a21*a33)
            g22 = (a11*a33-a31*a13)
            g23 = (a13*a21-a23*a11)

            g31 = (a21*a32-a22*a31)
            g32 = (a12*a31-a32*a11)
            g33 = (a11*a22-a21*a12)

            jacmq = a11*g11+a21*g12+a31*g13

            bmq(i,e)  = w3mq(i,1,1)*jacmq
            scale     = w3mq(i,1,1)/jacmq

            gf(1,i,e) = scale*(g11*g11+g12*g12+g13*g13) ! Grr
            gf(2,i,e) = scale*(g11*g21+g12*g22+g13*g23) ! Grs
            gf(3,i,e) = scale*(g11*g31+g12*g32+g13*g33) ! Grt
            gf(4,i,e) = scale*(g21*g21+g22*g22+g23*g23) ! Gss
            gf(5,i,e) = scale*(g21*g31+g22*g32+g23*g33) ! Gst
            gf(6,i,e) = scale*(g31*g31+g32*g32+g33*g33) ! Gtt

         enddo
      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine setprecn_bp1 (d,h1,h2)
C     Generate diagonal preconditioner for Helmholtz operator
C     Input: h1,h2                  Output: d
      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1,lg=3+3*(ldim-2))

      real*8    d(lx1,ly1,lz1,lelt),h1(lxyz,lelt),h2(lxyz,lelt)
      integer e

      real*8         gf(lg,lx1,ly1,lz1,lelt) ! Equivalence new gf() data
      equivalence (gf,g1m1)                  ! layout to g1m1...g6m1

      real*8 ysm1(ly1)

      nel   = nelfld(ifield)
      n     = nel*lx1*ly1*lz1
      nxyz  = lx1*ly1*lz1

      call copy    (d,bm1,n)   ! Mass matrix preconditioning full mass matrix
      call dssum   (d,nx1,ny1,nz1)
      call invcol1 (d,n)
      return

      call dsset(lx1,ly1,lz1)

      do 1000 e=1,nel

        call rzero(d(1,1,1,e),nxyz)

        do 320 iz=1,lz1
         do 320 iy=1,ly1
         do 320 ix=1,lx1
         do 320 iq=1,lx1
           d(ix,iy,iz,e) = d(ix,iy,iz,e)
     $                   + gf(1,iq,iy,iz,e) * dxm1(iq,ix)**2
     $                   + gf(2,ix,iq,iz,e) * dxm1(iq,iy)**2
     $                   + gf(3,ix,iy,iq,e) * dxm1(iq,iz)**2
  320    continue
C
C        Add cross terms if element is deformed.
C
         if (lxyz.gt.0) then

           do i2=1,ly1,ly1-1
           do i1=1,lx1,lx1-1
              d(1,i1,i2,e) = d(1,i1,i2,e)
     $            + gf(4,1,i1,i2,e) * dxtm1(1,1)*dytm1(i1,i1)
     $            + gf(5,1,i1,i2,e) * dxtm1(1,1)*dztm1(i2,i2)
              d(lx1,i1,i2,e) = d(lx1,i1,i2,e)
     $            + gf(4,lx1,i1,i2,e) * dxtm1(lx1,lx1)*dytm1(i1,i1)
     $            + gf(5,lx1,i1,i2,e) * dxtm1(lx1,lx1)*dztm1(i2,i2)
              d(i1,1,i2,e) = d(i1,1,i2,e)
     $            + gf(4,i1,1,i2,e) * dytm1(1,1)*dxtm1(i1,i1)
     $            + gf(6,i1,1,i2,e) * dytm1(1,1)*dztm1(i2,i2)
              d(i1,ly1,i2,e) = d(i1,ly1,i2,e)
     $            + gf(4,i1,ly1,i2,e) * dytm1(ly1,ly1)*dxtm1(i1,i1)
     $            + gf(6,i1,ly1,i2,e) * dytm1(ly1,ly1)*dztm1(i2,i2)
              d(i1,i2,1,e) = d(i1,i2,1,e)
     $            + gf(5,i1,i2,1,e) * dztm1(1,1)*dxtm1(i1,i1)
     $            + gf(6,i1,i2,1,e) * dztm1(1,1)*dytm1(i2,i2)
              d(i1,i2,lz1,e) = d(i1,i2,lz1,e)
     $            + gf(5,i1,i2,lz1,e) * dztm1(lz1,lz1)*dxtm1(i1,i1)
     $            + gf(6,i1,i2,lz1,e) * dztm1(lz1,lz1)*dytm1(i2,i2)

           enddo
           enddo
         endif

        do i=1,lxyz
           d(i,1,1,e)=d(i,1,1,e)*h1(i,e)+h2(i,e)*bm1(i,1,1,e)
        enddo

 1000 continue ! element loop

C     If axisymmetric, add a diagonal term in the radial direction (ISD=2)

      if (ifaxis.and.(isd.eq.2)) then
         do 1200 e=1,nel
            if (ifrzer(e)) call mxm(ym1(1,1,1,e),lx1,datm1,ly1,ysm1,1)
            k=0
            do 1190 j=1,ly1
            do 1190 i=1,lx1
               k=k+1
               if (ym1(i,j,1,e).ne.0.) then
                  term1 = bm1(i,j,1,e)/ym1(i,j,1,e)**2
                  if (ifrzer(e)) then
                     term2 =  wxm1(i)*wam1(1)*dam1(1,j)
     $                       *jacm1(i,1,1,e)/ysm1(i)
                  else
                     term2 = 0.
                  endif
                  d(i,j,1,e) = d(i,j,1,e) + h1(k,e)*(term1+term2)
               endif
 1190       continue
 1200    continue

      endif
      call dssum   (d,nx1,ny1,nz1)
      call invcol1 (d,n)

      if (nio.eq.0) write(6,1) n,d(1,1,1,1),h1(1,1),h2(1,1),bm1(1,1,1,1)
   1  format(i9,1p4e12.4,' diag prec')

      return
      end
C-----------------------------------------------------------------------
      subroutine setprecn_bp3 (d,h1,h2)
C     Generate dummy diagonal preconditioner for Helmholtz operator
C     Input: h1,h2                  Output: d
      include 'SIZE'
      include 'TOTAL'

      parameter (n=lx1*ly1*lz1*lelt)
      real*8 d(n),h1(n),h2(n)

      call rone (d,n)

      return
      end
C-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'TOTAL'

      if (istep.gt.0) call bp1

      return
      end
C-----------------------------------------------------------------------
      subroutine bp1
C     Solution to BP1 using libCEED
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'  ! ifsync
      include 'FDMH1'
      include 'ceedf.h'

      parameter (lzq=lx1+1)
      parameter (lx=lx1*ly1*lz1,lg=3+3*(ldim-2),lq=lzq**ldim)
      common /bpgfactors/ gf(lg*lq,lelt),bmq(lq,lelt),w3mq(lq)

      parameter (lt=lx1*ly1*lz1*lelt)
      parameter (ld=lxd*lyd*lzd*lelt)
      common /vcrns/ u1(lt),r1(lt),r2(lt),r3(lt)
      common /vcrny/ e1(lt)
      common /vcrvh/ h1(lt),h2(lx*lelt),pap(3)
      real*8 coords(ldim*lx*lelt)

      logical ifh3
      integer*8 ndof
      integer ceed,err,test
      character*64 spec

      integer p,q,ncomp,edof,ldof
      integer vec_p1,vec_ap1,vec_qdata,vec_coords,vec_rhs
      integer erstrctu,erstrctx,erstrctw
      integer basisu,basisx
      integer qf_mass,qf_setup
      integer op_mass,op_setup
      real*8  x,y,z

      external massf,masssetupf

      ifield = 1
      nxq    = nx1+1
      n = nx1*ny1*nz1*nelt

      ifsync = .false.

C     Set up coordinates
      ii=0
      do j=0,nelt-1
      do i=1,lx
        ii=ii+1
        x = xm1(ii,1,1,1)
        y = ym1(ii,1,1,1)
        z = zm1(ii,1,1,1)
        coords(i+0*lx+3*j*lx)=x
        coords(i+1*lx+3*j*lx)=y
        coords(i+2*lx+3*j*lx)=z
      enddo
      enddo

C     Init ceed library
      call get_spec(spec)
      call ceedinit(trim(spec)//char(0),ceed,err)

      call get_test(test)

C     Set up Nek geometry data
      call geodatq       (gf,bmq,w3mq,nxq)         ! Set up gf() arrays
      call set_h2_as_rhoJac_GL(h2,bmq,nxq)

C     Set up true soln
      call dist_fld_h1   (e1)
      call copy          (h1,e1,n)                 ! Save exact soln in h1

C     Set up solver parameters
      tol       = 1e-10
      param(22) = tol
      maxit     = 100

      call nekgsync()

C     Create ceed basis for mesh and computation
      p=nx1
      q=p+1
      ncomp=1
      call ceedbasiscreatetensorh1lagrange(ceed,ndim,ndim,p,q,
     $  ceed_gauss,basisx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,ndim,ncomp,p,q,
     $  ceed_gauss,basisu,err)

C     Create ceed element restrictions for mesh and computation
      edof=p**ldim
      ldof=edof*nelt*ncomp
      call ceedelemrestrictioncreateidentity(ceed,nelt,edof,ldof,
     $  ldim,erstrctx,err)
      call ceedelemrestrictioncreateidentity(ceed,nelt,edof,ldof,
     $  ncomp,erstrctu,err)
      call ceedelemrestrictioncreateidentity(ceed,nelt,q**ldim,
     $  nelt*q**ldim,1,erstrctw,err)

C     Create ceed vectors
      call ceedvectorcreate(ceed,ldof,vec_p1,err)
      call ceedvectorcreate(ceed,ldof,vec_ap1,err)
      call ceedvectorcreate(ceed,ldof,vec_rhs,err)
      call ceedvectorcreate(ceed,ldim*lx*nelt,vec_coords,err)
      call ceedvectorcreate(ceed,nelt*q**ldim,vec_qdata,err)

      call ceedvectorsetarray(vec_coords,ceed_mem_host,
     $  ceed_use_pointer,coords,err)

C     Create ceed qfunctions for masssetupf and massf
      call ceedqfunctioncreateinterior(ceed,1,masssetupf,
     $  __FILE__
     $  //':masssetupf',qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'x',ldim,
     $  ceed_eval_interp,err)
      call ceedqfunctionaddinput(qf_setup,'dx',ldim,
     $  ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_setup,'weight',1,
     $  ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,
     $  ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_setup,'rhs',1,
     $  ceed_eval_interp,err)

      call ceedqfunctioncreateinterior(ceed,1,massf,
     $  __FILE__
     $  //':massf',qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,
     $  ceed_eval_interp,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,
     $  ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,
     $  ceed_eval_interp,err)

C     Create ceed operators
      call ceedoperatorcreate(ceed,qf_setup,
     $  ceed_null,ceed_null,op_setup,err)
      call ceedoperatorsetfield(op_setup,'x',erstrctx,
     $  ceed_notranspose,basisx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'dx',erstrctx,
     $  ceed_notranspose,basisx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'weight',erstrctx,
     $  ceed_notranspose,basisx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'rho',erstrctw,
     $  ceed_notranspose,ceed_basis_collocated,
     $  ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rhs',erstrctu,
     $  ceed_notranspose,basisu,vec_rhs,err)

      call ceedoperatorcreate(ceed,qf_mass,
     $  ceed_null,ceed_null,op_mass,err)
      call ceedoperatorsetfield(op_mass,'u',erstrctu,
     $  ceed_notranspose,basisu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'rho',erstrctw,
     $  ceed_notranspose,ceed_basis_collocated,
     $  vec_qdata,err)
      call ceedoperatorsetfield(op_mass,'v',erstrctu,
     $  ceed_notranspose,basisu,ceed_vector_active,err)

C     Compute setup data
      call ceedvectorsetarray(vec_rhs,ceed_mem_host,
     $  ceed_use_pointer,r1,err)
      call ceedoperatorapply(op_setup,vec_coords,vec_qdata,
     $  ceed_request_immediate,err)
      write(6,*) 'After op_setup'
      call ceedvectorview(vec_rhs,err)

C     Set up true RHS
      call dssum         (r1,nx1,ny1,nz1)          ! r1

C     Set up algebraic RHS with libCEED
      call ceedvectorsetarray(vec_p1,ceed_mem_host,
     $  ceed_use_pointer,h1,err)
      call ceedvectorsetarray(vec_ap1,ceed_mem_host,
     $  ceed_use_pointer,r2,err)
      call ceedoperatorapply(op_mass,vec_p1,vec_ap1,
     $  ceed_request_immediate,err)                ! r2 = A_ceed*h1
      call dssum         (r2,nx1,ny1,nz1)

C     Set up algebraic RHS with Nek5000
      call axhm1         (pap,r3,h1,h1,h2,'bp1')   ! r3 = A_nek5k*h1
      call dssum         (r3,nx1,ny1,nz1)

      call nekgsync()

C     Solve true RHS
      if (nid.eq.0) write (6,*) "libCEED true RHS"
      tstart = dnekclock()
      call cggos(u1,r1,h1,h2,vmult,binvm1,tol,ceed,op_mass,
     $  vec_p1,vec_ap1,maxit,'bp1')
      tstop  = dnekclock()

C     Output
      telaps = (tstop-tstart)
      maxits = maxit
      er1 = glrdif(u1,e1,n)
      if (nid.eq.0) write(6,3) lx1,nelgv,er1,' error ',maxit

      if (test.eq.1.and.nid.eq.0) then
        if (maxit.gt.100) then
          write(6,*) "UNCONVERGED CG"
        endif
        if (dabs(er1).gt.5e-3) then
          write(6,*) "ERROR IS TOO LARGE"
        endif
      endif

      nx     = nx1-1
      ndof   = nelgt           ! ndofs
      ndof   = ndof*(nx**ldim) ! ndofs
      nppp   = ndof/np         ! ndofs/proc

      dofpss = ndof/telaps     ! DOF/sec - scalar form
      titers = telaps/maxits   ! time per iteration
      tppp_s = titers/nppp     ! time per iteraton per local point

      if (nid.eq.0) write(6,1) 'case scalar:'
     $ ,np,nx,nelt,nelgt,ndof,nppp,maxits,telaps,dofpss,titers,tppp_s

C     Solve libCEED algebraic RHS
      if (nid.eq.0) write (6,*) "libCEED algebraic RHS"
      maxit = 100
      tstart = dnekclock()
      call cggos(u1,r2,h1,h2,vmult,binvm1,tol,ceed,op_mass,
     $  vec_p1,vec_ap1,maxit,'bp1')
      tstop  = dnekclock()

C     Output
      telaps = (tstop-tstart)
      maxits = maxit
      er1 = glrdif(u1,e1,n)
      if (nid.eq.0) write(6,3) lx1,nelgv,er1,' error ',maxit

      if (test.eq.1.and.nid.eq.0) then
        if (maxit.gt.100) then
          write(6,*) "UNCONVERGED CG"
        endif
        if (dabs(er1).gt.1e-4) then
          write(6,*) "ERROR IS TOO LARGE"
        endif
      endif

      nx     = nx1-1
      ndof   = nelgt           ! ndofs
      ndof   = ndof*(nx**ldim) ! ndofs
      nppp   = ndof/np         ! ndofs/proc

      dofpss = ndof/telaps     ! DOF/sec - scalar form
      titers = telaps/maxits   ! time per iteration
      tppp_s = titers/nppp     ! time per iteraton per local point

      if (nid.eq.0) write(6,1) 'case scalar:'
     $ ,np,nx,nelt,nelgt,ndof,nppp,maxits,telaps,dofpss,titers,tppp_s

C     Solve Nek5000 algebraic RHS
      if (nid.eq.0) write (6,*) "Nek5000 algebraic RHS"
      maxit = 100
      tstart = dnekclock()
      call cggos(u1,r3,h1,h2,vmult,binvm1,tol,ceed,op_mass,
     $  vec_p1,vec_ap1,maxit,'bp1')
      tstop  = dnekclock()

C     Output
      telaps = (tstop-tstart)
      maxits = maxit
      er1 = glrdif(u1,e1,n)
      if (nid.eq.0) write(6,3) lx1,nelgv,er1,' error ',maxit

      if (test.eq.1.and.nid.eq.0) then
        if (maxit>=100) then
          write(6,*) "UNCONVERGED CG"
        endif
        if (dabs(er1)>1e-4) then
          write(6,*) "ERROR IS TOO LARGE"
        endif
      endif

      nx     = nx1-1
      ndof   = nelgt           ! ndofs
      ndof   = ndof*(nx**ldim) ! ndofs
      nppp   = ndof/np         ! ndofs/proc

      dofpss = ndof/telaps     ! DOF/sec - scalar form
      titers = telaps/maxits   ! time per iteration
      tppp_s = titers/nppp     ! time per iteraton per local point

      if (nid.eq.0) write(6,1) 'case scalar:'
     $ ,np,nx,nelt,nelgt,ndof,nppp,maxits,telaps,dofpss,titers,tppp_s

    1 format(a12,i7,i3,i7,i10,i14,i10,i4,1p4e13.5)
    3 format(i3,i9,e12.4,1x,a8,i9)

C     Destroy ceed handles
      call ceedvectordestroy(vec_p1,err)
      call ceedvectordestroy(vec_ap1,err)
      call ceedvectordestroy(vec_rhs,err)
      call ceedvectordestroy(vec_qdata,err)
      call ceedvectordestroy(vec_coords,err)
      call ceedelemrestrictiondestroy(erstrctu,err)
      call ceedelemrestrictiondestroy(erstrctx,err)
      call ceedelemrestrictiondestroy(erstrctw,err)
      call ceedbasisdestroy(basisu,err)
      call ceedbasisdestroy(basisx,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_mass,err)
      call ceeddestroy(ceed,err)

      return
      end
C-----------------------------------------------------------------------
      subroutine bp3
C     Solution to BP3 using libCEED
      include 'SIZE'
      include 'TOTAL'
      include 'CTIMER'  ! ifsync
      include 'FDMH1'
      include 'ceedf.h'

      parameter (lzq=lx1+1)
      parameter (lx=lx1*ly1*lz1,lg=3+3*(ldim-2),lq=lzq**ldim)
      common /bpgfactors/ gf(lg*lq,lelt),bmq(lq,lelt),w3mq(lq)

      parameter (lt=lx1*ly1*lz1*lelt)
      parameter (ld=lxd*lyd*lzd*lelt)
      common /vcrns/ u1(lt),r1(lt),r2(lt),r3(lt)
      common /vcrny/ e1(lt)
      common /vcrvh/ h1(lt),h2(ld),pap(3)
      real*8 coords(ldim*lx*lelt)

      logical ifh3
      integer*8 ndof
      integer ceed,err,test
      character*64 spec

      integer p,q,ncomp,edof,ldof
      integer vec_p1,vec_ap1,vec_qdata,vec_coords,vec_rhs
      integer erstrctu,erstrctx,erstrctw
      integer basisu,basisx
      integer qf_diffusion,qf_setup
      integer op_diffusion,op_setup
      integer ii,i,ngeo
      real*8  x,y,z

      external diffusionf,diffsetupf

      ifield = 1
      nxq    = nx1+1
      n = nx1*ny1*nz1*nelt

      ifsync = .false.

C     Set up coordinates and mask
      ii=0
      do j=0,nelt-1
      do i=1,lx
        ii=ii+1
        x = xm1(ii,1,1,1)
        y = ym1(ii,1,1,1)
        z = zm1(ii,1,1,1)
        coords(i+0*lx+3*j*lx)=x
        coords(i+1*lx+3*j*lx)=y
        coords(i+2*lx+3*j*lx)=z
        if ( x.eq.0.or.x.eq.1
     $   .or.y.eq.0.or.y.eq.1
     $   .or.z.eq.0.or.z.eq.1 ) then
          h2(ii)=0.
        else
          h2(ii)=1.
        endif
      enddo
      enddo

C     Init ceed library
      call get_spec(spec)
      call ceedinit(trim(spec)//char(0),ceed,err)

      call get_test(test)

C     Set up Nek geometry data
      call geodatq       (gf,bmq,w3mq,nxq)         ! Set up gf() arrays

C     Set up true soln
      call sin_fld_h1    (e1)
      call xmask1        (e1,h2,nelt)
      call copy          (h1,e1,n)                 ! Save exact soln in h1

C     Set up solver parameters
      tol       = 1e-10
      param(22) = tol
      maxit     = 100

      call nekgsync()

C     Create ceed basis for mesh and computation
      p=nx1
      q=p+1
      ncomp=1
      call ceedbasiscreatetensorh1lagrange(ceed,ldim,3*ncomp,p,q,
     $  ceed_gauss,basisx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,ldim,ncomp,p,q,
     $  ceed_gauss,basisu,err)

C     Create ceed element restrictions for mesh and computation
      edof=p**ldim
      ldof=edof*nelt*ncomp
      ngeo=(ldim*(ldim+1))/2
      call ceedelemrestrictioncreateidentity(ceed,nelt,edof,ldof,
     $  ldim,erstrctx,err)
      call ceedelemrestrictioncreateidentity(ceed,nelt,edof,ldof,
     $  ncomp,erstrctu,err)
      call ceedelemrestrictioncreateidentity(ceed,nelt,q**ldim,
     $  nelt*q**ldim,ngeo,erstrctw,err)

C     Create ceed vectors
      call ceedvectorcreate(ceed,ldof,vec_p1,err)
      call ceedvectorcreate(ceed,ldof,vec_ap1,err)
      call ceedvectorcreate(ceed,ldof,vec_rhs,err)
      call ceedvectorcreate(ceed,ldim*lx*nelt,vec_coords,err)
      call ceedvectorcreate(ceed,nelt*ngeo*q**ldim,vec_qdata,err)

      call ceedvectorsetarray(vec_coords,ceed_mem_host,
     $  ceed_use_pointer,coords,err)

C     Create ceed qfunctions for diffsetupf and diffusionf
      call ceedqfunctioncreateinterior(ceed,1,diffsetupf,
     $  __FILE__
     $  //':diffsetupf'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'x',ldim,
     $  ceed_eval_interp,err)
      call ceedqfunctionaddinput(qf_setup,'dx',ldim,
     $  ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_setup,'weight',1,
     $  ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',ngeo,
     $  ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_setup,'rhs',1,
     $  ceed_eval_interp,err)

      call ceedqfunctioncreateinterior(ceed,1,diffusionf,
     $  __FILE__
     $  //':diffusionf'//char(0),qf_diffusion,err)
      call ceedqfunctionaddinput(qf_diffusion,'u',1,
     $  ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_diffusion,'rho',ngeo,
     $  ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_diffusion,'v',1,
     $  ceed_eval_grad,err)  

C     Create ceed operators
      call ceedoperatorcreate(ceed,qf_setup,
     $  ceed_null,ceed_null,op_setup,err)
      call ceedoperatorsetfield(op_setup,'x',erstrctx,
     $  ceed_notranspose,basisx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'dx',erstrctx,
     $  ceed_notranspose,basisx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'weight',erstrctx,
     $  ceed_notranspose,basisx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'rho',erstrctw,
     $  ceed_notranspose,ceed_basis_collocated,
     $  ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rhs',erstrctu,
     $  ceed_notranspose,basisu,vec_rhs,err)

      call ceedoperatorcreate(ceed,qf_diffusion,
     $  ceed_null,ceed_null,op_diffusion,err)
      call ceedoperatorsetfield(op_diffusion,'u',erstrctu,
     $  ceed_notranspose,basisu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_diffusion,'rho',erstrctw,
     $  ceed_notranspose,ceed_basis_collocated,
     $  vec_qdata,err)
      call ceedoperatorsetfield(op_diffusion,'v',erstrctu,
     $  ceed_notranspose,basisu,ceed_vector_active,err)

C     Compute setup data
      call ceedvectorsetarray(vec_rhs,ceed_mem_host,
     $  ceed_use_pointer,r1,err)
      call ceedoperatorapply(op_setup,vec_coords,vec_qdata,
     $  ceed_request_immediate,err)

C     Set up true RHS
      call dssum         (r1,nx1,ny1,nz1)          ! r1
      call xmask1        (r1,h2,nelt)

C     Set up algebraic RHS with libCEED
      call ceedvectorsetarray(vec_p1,ceed_mem_host,
     $  ceed_use_pointer,h1,err)
      call ceedvectorsetarray(vec_ap1,ceed_mem_host,
     $  ceed_use_pointer,r2,err)
      call ceedoperatorapply(op_diffusion,vec_p1,vec_ap1,
     $  ceed_request_immediate,err)                ! r2 = A_ceed*h1
      call dssum         (r2,nx1,ny1,nz1)
      call xmask1        (r2,h2,nelt)

C     Set up algebraic RHS with Nek5000
      call axhm1         (pap,r3,h1,h1,h2,'bp3')   ! r3 = A_nek5k*h1
      call dssum         (r3,nx1,ny1,nz1)
      call xmask1        (r3,h2,nelt)

      call nekgsync()

C     Solve true RHS
      if (nid.eq.0) write (6,*) "libCEED true RHS"
      tstart = dnekclock()
      call cggos(u1,r1,h1,h2,vmult,binvm1,tol,ceed,op_diffusion,
     $  vec_p1,vec_ap1,maxit,'bp3')
      tstop  = dnekclock()

C     Output
      telaps = (tstop-tstart)
      maxits = maxit
      er1 = glrdif(u1,e1,n)
      if (nid.eq.0) write(6,3) lx1,nelgv,er1,' error ',maxit

      if (test.eq.1.and.nid.eq.0) then
        if (maxit>=100) then
          write(6,*) "UNCONVERGED CG"
        endif
        if (dabs(er1)>1e-4) then
          write(6,*) "ERROR IS TOO LARGE"
        endif
      endif

      nx     = nx1-1
      ndof   = nelgt           ! ndofs
      ndof   = ndof*(nx**ldim) ! ndofs
      nppp   = ndof/np         ! ndofs/proc

      dofpss = ndof/telaps     ! DOF/sec - scalar form
      titers = telaps/maxits   ! time per iteration
      tppp_s = titers/nppp     ! time per iteraton per local point

      if (nid.eq.0) write(6,1) 'case scalar:'
     $ ,np,nx,nelt,nelgt,ndof,nppp,maxits,telaps,dofpss,titers,tppp_s

C     Solve libCEED algebraic RHS
      if (nid.eq.0) write (6,*) "libCEED algebraic RHS"
      maxit = 100
      tstart = dnekclock()
      call cggos(u1,r2,h1,h2,vmult,binvm1,tol,ceed,op_diffusion,
     $  vec_p1,vec_ap1,maxit,'bp3')
      tstop  = dnekclock()

C     Output
      telaps = (tstop-tstart)
      maxits = maxit
      er1 = glrdif(u1,e1,n)
      if (nid.eq.0) write(6,3) lx1,nelgv,er1,' error ',maxit

      if (test.eq.1.and.nid.eq.0) then
        if (maxit>=100) then
          write(6,*) "UNCONVERGED CG"
        endif
        if (dabs(er1)>1e-4) then
          write(6,*) "ERROR IS TOO LARGE"
        endif
      endif

      nx     = nx1-1
      ndof   = nelgt           ! ndofs
      ndof   = ndof*(nx**ldim) ! ndofs
      nppp   = ndof/np         ! ndofs/proc

      dofpss = ndof/telaps     ! DOF/sec - scalar form
      titers = telaps/maxits   ! time per iteration
      tppp_s = titers/nppp     ! time per iteraton per local point

      if (nid.eq.0) write(6,1) 'case scalar:'
     $ ,np,nx,nelt,nelgt,ndof,nppp,maxits,telaps,dofpss,titers,tppp_s

C     Solve Nek5000 algebraic RHS
      if (nid.eq.0) write (6,*) "Nek5000 algebraic RHS"
      maxit = 100
      tstart = dnekclock()
      call cggos(u1,r3,h1,h2,vmult,binvm1,tol,ceed,op_diffusion,
     $  vec_p1,vec_ap1,maxit,'bp3')
      tstop  = dnekclock()

C     Output
      telaps = (tstop-tstart)
      maxits = maxit
      er1 = glrdif(u1,e1,n)
      if (nid.eq.0) write(6,3) lx1,nelgv,er1,' error ',maxit

      if (test.eq.1.and.nid.eq.0) then
        if (maxit>=100) then
          write(6,*) "UNCONVERGED CG"
        endif
        if (dabs(er1)>1e-4) then
          write(6,*) "ERROR IS TOO LARGE"
        endif
      endif

      nx     = nx1-1
      ndof   = nelgt           ! ndofs
      ndof   = ndof*(nx**ldim) ! ndofs
      nppp   = ndof/np         ! ndofs/proc

      dofpss = ndof/telaps     ! DOF/sec - scalar form
      titers = telaps/maxits   ! time per iteration
      tppp_s = titers/nppp     ! time per iteraton per local point

      if (nid.eq.0) write(6,1) 'case scalar:'
     $ ,np,nx,nelt,nelgt,ndof,nppp,maxits,telaps,dofpss,titers,tppp_s

    1 format(a12,i7,i3,i7,i10,i14,i10,i4,1p4e13.5)
    3 format(i3,i9,e12.4,1x,a8,i9)

C     Destroy ceed handles
      call ceedvectordestroy(vec_p1,err)
      call ceedvectordestroy(vec_ap1,err)
      call ceedvectordestroy(vec_rhs,err)
      call ceedvectordestroy(vec_qdata,err)
      call ceedvectordestroy(vec_coords,err)
      call ceedelemrestrictiondestroy(erstrctu,err)
      call ceedelemrestrictiondestroy(erstrctx,err)
      call ceedelemrestrictiondestroy(erstrctw,err)
      call ceedbasisdestroy(basisu,err)
      call ceedbasisdestroy(basisx,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_diffusion,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_diffusion,err)
      call ceeddestroy(ceed,err)

      return
      end
C-----------------------------------------------------------------------
      subroutine cggos(u1,r1,h1,h2,rmult,binv,tin,ceed,ceed_op,vec_p1,
     $  vec_ap1,maxit,bpname)
C     Scalar conjugate gradient iteration for solution of uncoupled
C     Helmholtz equations
C     Input: r1,h1,h2,rmult,binv,tin,ceed,ceed_op,vec_p1,vec_ap1,bpname
C     Output: u1,maxit
      include 'SIZE'
      include 'TOTAL'
      include 'DOMAIN'
      include 'FDMH1'
      character*3 bpname

C     INPUT:  rhs1 - rhs
C             h1   - exact solution

      parameter (lt=lx1*ly1*lz1*lelt)
      parameter (ld=lxd*lyd*lzd*lelt)
      real*8 u1(lt),r1(lt),h1(lt),h2(lt)
      real*8 rmult(1),binv(1)
      integer ceed,ceed_op,vec_ap1,vec_p1
      common /scrcg/ dpc(lt),p1(lt),z1(lt)
      common /scrca/ wv(4),wk(4),rpp1(4),rpp2(4),alph(4),beta(4),pap(4)

      real*8 ap1(lt)
      equivalence (ap1,z1)

      vol   = volfld(ifield)
      nel   = nelfld(ifield)
      nxyz  = lx1*ly1*lz1
      n     = nxyz*nel
      nx    = nx1-1                  ! Polynomial order (just for i/o)

      tol=tin

      if(bpname.ne.'bp1') then
        call setprecn_bp3(dpc,h1,h2) ! Set up diagional pre-conidtioner
      else
        call setprecn_bp1(dpc,h1,h2) ! Set up diagional pre-conidtioner
      endif

      call rzero         (u1,n)      ! Initialize solution

      wv(1)=0
      do i=1,n
         s=rmult(i)                  !      -1
         p1(i)=dpc(i)*r1(i)          ! p = M  r      T
         wv(1)=wv(1)+s*p1(i)*r1(i)   !              r p
      enddo
      call gop(wv(1),wk,'+  ',1)
      rpp1(1) = wv  (1)

      do 1000 iter=1,maxit
         call axhm1_ceed (pap,ap1,p1,h1,h2,ceed,ceed_op,
     $     vec_ap1,vec_p1)
         call dssum    (ap1,nx1,ny1,nz1)
         if (bpname.ne.'bp1') call xmask1(ap1,h2,nel)

         call gop      (pap,wk,'+  ',1)
         alph(1) = rpp1(1)/pap(1)

         do i=1,n
            u1(i)=u1(i)+alph(1)* p1(i)
            r1(i)=r1(i)-alph(1)*ap1(i)
         enddo

C        tolerance check here
         call rzero(wv,2)
         do i=1,n
            wv(1)=wv(1)+r1(i)*r1(i)            ! L2 error estimate
            z1(i)=dpc(i)*r1(i)                 ! z = M  r
            wv(2)=wv(2)+rmult(i)*z1(i)*r1(i)   ! r z
         enddo
         call gop(wv,wk,'+  ',2)

C        if (nio.eq.0) write(6,1) ifield,istep,iter,nx,(wv(k),k=1,1)
  1     format(i2,i9,i5,i4,1p1e12.4,' cggos')

         enorm=sqrt(wv(1))
         if (enorm.lt.tol) then
            ifin = iter
            if (nio.eq.0) write(6,3000) istep,ifin,enorm,tol
            goto 9999
         endif
C        if (nio.eq.0) write(6,2) iter,enorm,alph(1),pap(1),'alpha'
 2      format(i5,1p3e12.4,2x,a5)

         rpp2(1)=rpp1(1)
         rpp1(1)=wv  (2)
         beta1  =rpp1(1)/rpp2(1)
         do i=1,n
            p1(i)=z1(i) + beta1*p1(i)
         enddo

 1000 continue

      rbnorm=sqrt(wv(1))
      if (nio.eq.0) write (6,3001) istep,iter,rbnorm,tol
      iter = iter-1

 9999 continue

      maxit=iter

 3000 format(i12,1x,'cggo scalar:',i6,1p5e13.4)
 3001 format(2i6,' Unconverged cggo scalar: rbnorm =',1p2e13.6)

      return
      end
C-----------------------------------------------------------------------
      subroutine axhm1_ceed(pap,ap1,p1,h1,h2,ceed,ceed_op,
     $  vec_ap1,vec_p1)
C     Vector conjugate gradient matvec for solution of uncoupled
C     Helmholtz equations
C     Input: pap,p1,h1,h2,bpname,ceed,ceed_op,vec_ap1,vec_p1
C     Output: ap1
      include 'SIZE'
      include 'TOTAL'
      include 'ceedf.h'

      parameter (lx=lx1*ly1*lz1,lg=3+3*(ldim-2))
      real*8       gf(lg,lx,lelt)             ! Equivalence new gf() data
      equivalence (gf,g1m1)                   ! layout to g1m1...g6m1

      real*8   pap(3)
      real*8   ap1(lx,lelt)
      real*8    p1(lx,lelt)
      real*8    h1(lx,lelt),h2(lx,lelt)
      integer ceed,ceed_op,vec_ap1,vec_p1,err
      integer i,e
C TODO replace this by SyncArray when available
      integer*8 offset

      call ceedvectorsetarray(vec_p1,ceed_mem_host,ceed_use_pointer,
     $  p1,err)
      call ceedvectorsetarray(vec_ap1,ceed_mem_host,ceed_use_pointer,
     $  ap1,err)

      call ceedoperatorapply(ceed_op,vec_p1,vec_ap1,
     $  ceed_request_immediate,err)

C TODO replace this by SyncArray when available
      call ceedvectorgetarrayread(vec_ap1,ceed_mem_host,ap1,offset,err)
      call ceedvectorrestorearrayread(vec_ap1,ap1,offset,err)

      pap(1)=0.

      do e=1,nelt
         do i=1,lx
           pap(1)=pap(1)+ap1(i,e)*p1(i,e)
         enddo
      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine ax_e_bp1(w,u,g,h1,h2,b,ju,us,ut)
C     Local matrix-vector for solution of BP3 (stiffness matrix)
C     Input: u,g,h1,h2,b,ju,us,ut   Output: w
      include 'SIZE'
      include 'TOTAL'

      parameter (lxyz=lx1*ly1*lz1,lg=3+3*(ldim-2))
      real*8 w(lxyz),u(lxyz),g(lg,lxyz),h1(lxyz),h2(lxyz),b(lxyz)
      real*8 ju(lxyz),us(lxyz),ut(lxyz)

      nxq = nx1+1 ! Number of quadrature points

      lxyzq = nxq**ldim

      call intp_rstd (ju,u,lx1,nxq,if3d,0) ! 0 --> Fwd interpolation
      do i=1,lxyzq
         ju(i)=ju(i)*h2(i) !! h2 must be on the fine grid, w/ quad wts
      enddo
      call intp_rstd (w,ju,lx1,nxq,if3d,1) ! 1 --> ju-->u

      return
      end
C-----------------------------------------------------------------------
      subroutine axhm1_bp1(pap,ap1,p1,h1,h2)
C     Vector conjugate gradient matvec for solution of BP1 (mass matrix)
C     Input: pap,p1,h1,h2           Output: ap1
      include 'SIZE'
      include 'TOTAL'

      parameter (lx=lx1*ly1*lz1,lg=3+3*(ldim-2))
      real*8         gf(lg,lx,lelt)             ! Equivalence new gf() data
      equivalence (gf,g1m1)                     ! layout to g1m1...g6m1

      real*8 pap(3)
      real*8 ap1(lx,lelt)
      real*8  p1(lx,lelt)
      real*8  h1(lx,lelt),h2(lx,lelt)

      real*8 ur(lx),us(lx),ut(lx)
      common /ctmp1/ ur,us,ut

      integer e

      pap(1)=0.

      k=1
      nxq = nx1+1

      do e=1,nelt

         call ax_e_bp1(ap1(1,e),p1(1,e),gf(1,1,e),h1(1,e),h2(k,1)
     $                                          ,bm1(1,1,1,e),ur,us,ut)
         do i=1,lx
           pap(1)=pap(1)+ap1(i,e)*p1(i,e)
         enddo
         k=k+nxq*nxq*nxq

      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine ax_e_bp3(w,u,g,ur,us,ut,wk)
C     Local matrix-vector for solution of BP3 (stiffness matrix)
C     Input: u,g,ur,us,ut,wk        Output: w
      include 'SIZE'
      include 'TOTAL'

      parameter (lzq=lx1+1,lxyz=lx1*lx1*lx1,lxyzq=lzq*lzq*lzq)

      common /ctmp0/ tmp(lxyzq)
      common /dxmfine/ dxmq(lzq,lzq),dxtmq(lzq,lzq)

      real*8 ur(lxyzq),us(lxyzq),ut(lxyzq),wk(lxyzq)
      real*8 w(lxyz),u(lxyz),g(2*ldim,lxyzq)

      n = lzq-1

      call intp_rstd  (wk,u,lx1,lzq,if3d,0) ! 0 --> Fwd interpolation
      call loc_grad3  (ur,us,ut,wk,n,dxmq,dxtmq)

      do i=1,lxyzq
         wr = g(1,i)*ur(i) + g(2,i)*us(i) + g(3,i)*ut(i)
         ws = g(2,i)*ur(i) + g(4,i)*us(i) + g(5,i)*ut(i)
         wt = g(3,i)*ur(i) + g(5,i)*us(i) + g(6,i)*ut(i)
         ur(i) = wr
         us(i) = ws
         ut(i) = wt
      enddo

      call loc_grad3t (wk,ur,us,ut,n,dxmq,dxtmq,tmp)
      call intp_rstd  (w,wk,lx1,lzq,if3d,1) ! 1 --> ju-->u

      return
      end
C-----------------------------------------------------------------------
      subroutine axhm1_bp3(pap,ap1,p1,h1,h2)
C     Vector conjugate gradient matvec for solution of BP3 (stiffness matrix)
C     Input: pap,p1,h1,h2           Output: ap1
      include 'SIZE'
      include 'TOTAL'

      parameter (lzq=lx1+1)
      parameter (lx=lx1*ly1*lz1,lg=3+3*(ldim-2),lq=lzq**ldim)
      common /bpgfactors/ gf(lg,lq,lelt),bmq(lq,lelt),w3mq(lq)

      real*8 pap(3)
      real*8 ap1(lx,lelt)
      real*8  p1(lx,lelt)
      real*8  h1(lx,lelt),h2(lx,lelt)

      common /ctmp1/ ur,us,ut,wk
      real*8 ur(lq),us(lq),ut(lq),wk(lq)

      integer e

      pap(1)=0.

      do e=1,nelt

         call ax_e_bp3(ap1(1,e),p1(1,e),gf(1,1,e),ur,us,ut,wk)
         do i=1,lx
           pap(1)=pap(1)+p1(i,e)*ap1(i,e)
         enddo

      enddo

      return
      end
C-----------------------------------------------------------------------
      subroutine axhm1(pap,ap1,p1,h1,h2,bpname)
C     Vector conjugate gradient matvec for solution of uncoupled
C     Helmholtz equations
C     Input: pap,p1,h1,h2,bpname    Output: ap1
      include 'SIZE'
      include 'TOTAL'

      parameter (lx=lx1*ly1*lz1)

      real*8 pap(3),ap1(lx,lelt),p1(lx,lelt)
      real*8 h1(lx,lelt),h2(lx,lelt)

      character*3 bpname

      if (bpname.eq.'bp1') then
         call axhm1_bp1(pap,ap1,p1,h1,h2)

      elseif (bpname.eq.'bp3') then
         call axhm1_bp3(pap,ap1,p1,h1,h2)

      else
         write(6,*) bpname,' axhm1 bpname error'
         stop

      endif

      return
      end
C-----------------------------------------------------------------------
      subroutine get_spec(spec)
C     Get CEED backend specification
C     Input:                        Output: spec
      integer i
      character*64 spec

      spec = '/cpu/self'
      if(iargc().ge.1) then
        call getarg(1,spec)
        write(6,*) spec
      endif

      return
      end
C-----------------------------------------------------------------------
      subroutine get_test(test)
C     Get test mode flag
C     Input:                        Output: test
      integer i,test
      character*64 testval

      test=0
      if(iargc().ge.2) then
        call getarg(2,testval)
      endif
      if(testval.eq."test") then
        test=1
      endif

      return
      end
C-----------------------------------------------------------------------

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end

c automatically added by makenek
      subroutine userqtl

      call userqtl_scig

      return
      end
