using ITensors
using ITensorTDVP
using DelimitedFiles

# Lets clear the code a little bit in order to make it easier to distribute the jobs and make all the process more efficient
# [X] comment the code so I can remember well how it works
# [X] implement a new function that allows me to differentiate if Im above or under the Fermi energy
# [X] implement the rule of thumb for time propagation (this one is tricky and the parameters have to be rechecked)
# [X] do a bash code that is able to create the carpets, execute the code and write the results in an order fashion
# [ ] leave everything running in Darwin/Keldysh/Kohn
#
# Now that, apparently, everything is more or less done the execution of the code has a very strict syntaxis
# In order to execute the code one must type julia *.jl [SO INT VALUE] [START OF GATEV] [END OF GATEV] [LENGTH OF GATEV] [FLAG] [FILENAME]

function H_build(Nx, t, U, Jex, Up, JH, γ, α, J, w)
  """
  H_build: constructs the  hamiltonian as an MPO
  Nx: number of sites in the chain
  t: hopping parameter (tipycally set to 1 in order to set the scale)
  U: onsite repulsion
  Jex: Heisenberg-like term
  Up: One of Hunds terms
  JH: Another of the Hunds parameters
  gamma: final Hund parameter
  J: Probe-system interaction, has to be small compared to the other parameters if high resolution of the spectral function is to be achieved
  w: gate voltage (this is the term that is varied when the spectral function is calcualted)
  """
  os = OpSum()
  for n in 1:(Nx-1)
    # Hopping interaction for the first orbital
    os += -t, "Cdagup", 3*n-2, "Cup", 3*n+1
    os += -t, "Cdagup", 3*n+1, "Cup", 3*n-2
    os += -t, "Cdagdn", 3*n-2, "Cdn", 3*n+1
    os += -t, "Cdagdn", 3*n+1, "Cdn", 3*n-2
    os += -Jex, "Sz", 3*n-2, "Sz", 3*n+1
    os += -0.5*Jex, "S-", 3*n-2, "S+", 3*n+1
    os += -0.5*Jex, "S+", 3*n-2, "S-", 3*n+1

    # Rashba parameters
    os += -0.5*α, "Cdagup", 3*n-2, "Cdn", 3*n+1
    os += -0.5*α, "Cdagdn", 3*n+1, "Cup", 3*n-2
    os += 0.5*α, "Cdagdn", 3*n-2, "Cup", 3*n+1
    os += 0.5*α, "Cdagup", 3*n+1, "Cdn", 3*n-2
  end

  for n in 1:(Nx-1)
    # Hopping interaction for the second orbital
    os += -t, "Cdagup", 3*n, "Cup", 3*n+3
    os += -t, "Cdagup", 3*n+3, "Cup", 3*n
    os += -t, "Cdagdn", 3*n, "Cdn", 3*n+3
    os += -t, "Cdagdn", 3*n+3, "Cdn", 3*n
    os += -Jex, "Sz", 3*n, "Sz", 3*n+3
    os += -0.5*Jex, "S-", 3*n, "S+", 3*n+3
    os += -0.5*Jex, "S+", 3*n, "S-", 3*n+3

    # Rashba parameters
    os += -0.5*α, "Cdagup", 3*n, "Cdn", 3*n+3
    os += -0.5*α, "Cdagdn", 3*n+3, "Cup", 3*n
    os += 0.5*α, "Cdagdn", 3*n, "Cup", 3*n+3
    os += 0.5*α, "Cdagup", 3*n+3, "Cdn", 3*n

   end

  # Hund terms
  for n in 1:Nx
    os += Up-JH, "Nup", 3*n-2, "Nup", 3*n
    os += Up-JH, "Ndn", 3*n-2, "Ndn", 3*n
    os += Up, "Nup", 3*n-2, "Ndn", 3*n
    os += Up, "Ndn", 3*n-2, "Nup", 3*n
    os += γ*JH, "H1", 3*n-2, "H1dag", 3*n
    os += γ*JH, "H1", 3*n, "H1dag", 3*n-2
    os += γ*JH, "H2", 3*n-2, "H2dag", 3*n
    os += γ*JH, "H2", 3*n, "H2dag", 3*n-2
  end

  # On-site repulsion
  for n in 1:Nx
    os += U, "Nupdn", 3*n
    os += U, "Nupdn", 3*n-2
  end
  
  # Probe-internal orbital coupling
  for n in 1:Nx
    os += J, "Cdagup", 3*n-1, "Cup", 3*n
    os += J, "Cdagup", 3*n, "Cup", 3*n-1
  end

  # Gate Voltage terms
  for n in 1:Nx
    os += w, "Nup", 3*n-1
    os += 100, "Ndn", 3*n-1
  end
 
  return os
end

function Compute_spectral(C, Nx, k)
  # Compute momentum resolved spectral function from the measurements in the probe virtual sites
  a=0  
  for i in 1:Nx
    for j in 1:Nx
      a += C[3*i-1,3*j-1]*exp(im*k*(i-j)) 
    end
  end
  a /= Nx
  return real(a)
end

function Compute_seg(C, Nx, kmin, kmax, kstep)
  # Iter momentum resolved spectral function for several momentum values
  K=range(kmin,stop=kmax,length=kstep)
  X = zeros(length(K))
  @show X
  iter=1
  for i in K
    X[iter]=Compute_spectral(C,Nx,i)
    iter+=1
  end
  return X
end

# New operators needed for the Hund terms

ITensors.op(::OpName"H1",::SiteType"Electron")=
  [
     0.0 0.0 0.0 0.0
     0.0 0.0 0.0 0.0
     0.0 0.0 0.0 0.0
     1.0 0.0 0.0 0.0
  ]

ITensors.op(::OpName"H1dag",::SiteType"Electron")=
  [
     0.0 0.0 0.0 1.0
     0.0 0.0 0.0 0.0
     0.0 0.0 0.0 0.0
     0.0 0.0 0.0 0.0
  ]

ITensors.op(::OpName"H2",::SiteType"Electron")=
  [
     0.0 0.0 0.0 0.0
     0.0 0.0 1.0 0.0
     0.0 0.0 0.0 0.0
     0.0 0.0 0.0 0.0
  ]

ITensors.op(::OpName"H2dag",::SiteType"Electron")=
  [
     0.0 0.0 0.0 0.0
     0.0 0.0 0.0 0.0
     0.0 1.0 0.0 0.0
     0.0 0.0 0.0 0.0
  ]

function Run(Ny, Nx, t, U, Jex, Up, JH, γ, α, J, w, flag, filename)
    """
    The idea is that this new Run function avoids the problems of manually changing some lines depending of the calculation is done above or under the Fermi level
    The only new and relevant parameter is the 'flag' parameter that indicates if the calculation is done over or under the Fermi level
    """
    if flag==true
        winit=100
    else
        winit=-100
    end
    N = Nx * Ny
    #Definition of the site type
    sites = siteinds("Electron", N; conserve_qns=true, conserve_sz=false)
    #Definition of the lattice
    lattice = square_lattice(Nx, Ny; yperiodic = false)

    # Define the Hund Hamiltonian on this lattice
    H = MPO(H_build(Nx,t,U,Jex,Up,JH,γ,α,J,winit),sites)
    state = ["Emp" for n in 1:N]
    if flag==true
        for i in 1:Nx
            state[3*i] = (isodd(i) ? "Up" : "Dn")
            state[3*i-2] = (isodd(i) ? "Dn" : "Up")
        end
    else
        for i in 1:Nx
            state[3*i] = (isodd(i) ? "Up" : "Dn")
            state[3*i-2] = (isodd(i) ? "Dn" : "Up")
            state[3*i-1] = "Up"
        end
    end
    psi0 = randomMPS(sites,state,10)

    # In this case I choose fixed DMRG parameters rather thanintroducing them as further arguments to the function
    sweeps = Sweeps(60)
    maxdim!(sweeps,100,200,200,400,400,800,1200,1200,1200,1600,1600,2000)
    noise!(sweeps,1E-5,1E-6,1E-6,1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-6, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-7, 1E-8,1E-9,0)
    cutoff!(sweeps,1E-8)
    @show sweeps

    # DMRG calculation
    energy,psi = dmrg(H,psi0,sweeps)

    ttotal = 6
    tau = 0.1
    cutoff = 1e-6
    maxdim = 700

    for V_g in w
        H1 = MPO(H_build(Nx,t,U,Jex,Up,JH,γ,α,J,V_g),sites)

        println(V_g)

        # Real time evolution through TDVP algorithm
        psi_f = tdvp(H1,
             -im * ttotal,
             psi;
             time_step= -im*tau,
             cutoff,
	     maxdim,
	     outputlevel=1,
             normalize=true,
	     )

        # computation of the correlation matrix
        if flag==true
            c= correlation_matrix(psi_f,"Cdagup","Cup")
        else
            c= correlation_matrix(psi_f,"Cup", "Cdagup")
        end
  
        # computation of the spectral function
        A = Compute_seg(c, Nx, -pi, pi, 100)

        filename1=filename*string(V_g)*".txt"
        writedlm(filename1,A)  
    end
    return
end

function Create_w(vinit, vend, length)
    return range(vinit, stop=vend, length=length)
end

  
# Parameters of the Hamiltonian
# in this case we are considering a 3x24 quasi 1D geometry, the first two layers are the system of interest and the Jex value is a little bit low to appretiated
Ny = 3
Nx = 24
t = 1
U = 8
Jex = -1
Up = 6
JH = 1
gamma = 1
alpha = parse(Int64,ARGS[1])

# Parameters of the probe part of the system
J = 0.2
w = Create_w(parse(Int64,ARGS[2]), parse(Int64,ARGS[3]), parse(Int64,ARGS[4]))
@show w
flag = parse(Int64,ARGS[5])
filename = ARGS[6]
Run(Ny, Nx, t, U, Jex, Up, JH, gamma, alpha, J, w, flag, filename)

