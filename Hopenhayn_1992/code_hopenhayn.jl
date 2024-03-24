#======================================================================================================#
# Solving the Hopenhayn (1992) model with VFI + Bisection Method 
# Author: Samiran Dutta (based on codes by Tomás R. Martinez (https://tomasrm.github.io))
# Date: 17.03.24
#======================================================================================================#

# Use Pkg.add(.) lines if packages are not already installed
using Parameters 
using Printf
using Distributions
using Roots
using Plots
using LinearAlgebra
using SparseArrays
using LaTeXStrings
using Statistics
using Pkg
using PGFPlotsX #it's heavy, use only if you want the figures to be reproduced using TikZpicture, make sure Julia is able to find your LaTeX installation for this! 

@time begin

#======================================================================================================#
# Parameters + Discretization
#======================================================================================================#

#Tauchen Function to discretize continuous process 
function tauchen(N, ρ, σ; μ = 0.0, m = 3.0)
	s1    = μ/(1-ρ) - m*sqrt(σ^2/(1-ρ^2))
   	sN    = μ/(1-ρ) + m*sqrt(σ^2/(1-ρ^2))
    s = collect(range(s1, sN, length = N))
    step    = (s[N]-s[1])/(N-1)  #evenly spaced grid
    P      = fill(0.0, N, N)

    for i = 1:ceil(Int, N/2)
    	P[i, 1] = cdf.(Normal(), (s[1] - μ - ρ*s[i] + step/2.0)/σ)
        P[i, N]  = 1 - cdf.(Normal(), (s[N] - μ - ρ*s[i]  - step/2.0)/σ)
        for j = 2:N-1
        	P[i,j]  = cdf.(Normal(), (s[j] - μ - ρ*s[i]  + step/2.0)/σ) -
                            cdf.(Normal(), (s[j] - μ - ρ*s[i] - step/2.0)/σ)
        end
        P[floor(Int, (N-1)/2+2):end, :]=P[ceil(Int ,(N-1)/2):-1:1, end:-1:1]
	end

    ps = sum(P, dims = 2)
    P = P./ps

    return s, P
end

#Setting parameters 
#Note: AR(1) process for productivity: log(ϕ) = κ + ρ log(ϕ) + σ.ϵ ; where ϵ ≈ N(0,1) 
#Production function: y = ϕ n^α
function set_par(;
    β = 0.8,                           #Firm's discount factor β = 1/(1+r)
    ρ = 0.9,                           #Persistence of AR(1)
    σ = 0.2,                           #Volatility of AR(1)
    ϕ_mean = 1.39,                     #Mean of AR(1)
    α = 2/3,                           #Income share of labor 
    c_e = 40,                          #Entry cost 
    c_f = 20,                          #Per-period fixed cost 
    D_bar = 100.0,                     #Invariant demand parameter                    
    n_ϕ = 101,                         #Number of grid points for state ϕ
    w = 1.0)                           #Taking wage as numeraire 

    #Discretization 
    κ = ϕ_mean*(1-ρ) 
    gPhi, F_trans = tauchen(n_ϕ, ρ, σ; μ = κ, m = 4.0) 
    gPhi = @. exp(gPhi)
    invPhi = F_trans^1000 # invariant distribution
	invPhi = invPhi[1,:]	

    #ENTRANTS DISTRIBUTION: Assume they draw from the invariant distribution.
	G_prob = invPhi

    return (β = β, F_trans = F_trans, gPhi = gPhi, α = α, c_e = c_e, 
            c_f = c_f, D_bar = D_bar, n_ϕ = n_ϕ, w = w, G_prob = G_prob)
end

param = set_par()
#@unpack n_ϕ, gPhi, F_trans, G_prob = param             #to check stuff

#======================================================================================================#
# Solving the Bellman equation (VFI)
#======================================================================================================#
function solve_bellman(p_guess, param)
    @unpack β, F_trans, gPhi, α, c_f, n_ϕ, w = param

    #Static decision
    gN = @. ((p_guess*α*gPhi)/w)^(1/(1-α))
    gΠ = @. p_guess*gPhi*gN^α - w*gN - c_f*w             #Note: the fixed cost is paid in terms of workers so c_f*w, but w=1 so doesn't really matter..

    #bellman parameters
    tol = 10.0^-9
    max_iter = 500

    #Initialising V
    V = zeros(n_ϕ)                                   

    #VFI
    function vfi_func(V)
        v_guess = zeros(n_ϕ)                             #Initialise v_guess
        iter = 0 

        while iter <= max_iter
            iter += 1
            println("Iteration #$iter")
            v_guess = copy(V)                                

            V = gΠ + β*max.(F_trans*v_guess, 0)           #Note: F_trans*V is (n_ϕ x n_ϕ) * (n_ϕ x 1) = n_ϕ x 1

            sup = maximum(abs.((V - v_guess)/V))          #Taking the percentage difference [Note: max() is used to find the max value between two arguments, while maximum() finds the max val in an array]

            if sup < tol 
                println("Value function converged! Max. % difference = $sup")
                break
            end 

            if iter == max_iter 
                println("Max iterations achieved, VF did not converge! Max. difference = $sup")
            end

        end 
    
        χ = zeros(n_ϕ);
        χ[F_trans*V .< 0.0] .= 1.0;                       #Those with a negative expected continuation value => exit (χ=1)
        χ[F_trans*V .>= 0.0] .= 0.0;                      #Those with a positive expected continuation value => stay (χ=0)

        return V, χ 
    end

    V, χ = vfi_func(V)

  return (V, χ, gN, gΠ)       
  
end 

#sol_bel = solve_bellman(2, param)                        #Uncomment to check VFI

#======================================================================================================#
# Solving for the equilibrium price (Bisection Method)
#======================================================================================================#

function find_eqprice(param)

    @unpack c_e, β, G_prob, w = param                                     

    p_min, p_max = 1, 5       #some random guess
    max_iter_p = 500
    iter_p = 0
    p_star = 0.0              # Initialize price 

    while iter_p <= max_iter_p
        iter_p += 1
        p_guess = (p_min + p_max)/2 

        V, χ, gN, gΠ = solve_bellman(p_guess, param)
        V_e = -c_e*w + β*sum(G_prob.*V) 

        println("Iteration #", iter_p, "; V_e = ", V_e)

        if abs(V_e) < 1e-5
            p_star = p_guess
            println("Equilibrium price found! Equilibrium Price = ", p_guess, ", now iterating on Bellman with this price:")  
            break
        elseif V_e > 0 
            p_max = p_guess
            println("Excess entry, setting p_guess = ", (p_min + p_max)/2, "; p_max = ", p_max, " , p_min = ", p_min)
        elseif V_e < 0
            p_min = p_guess
            println("No entry, setting p_guess = ", (p_min + p_max)/2, "; p_max = ", p_max, " , p_min = ", p_min)
        end

    end

    V, χ, gN, gΠ = solve_bellman(p_star, param)

    return (p_star = p_star, V = V, χ = χ, gN = gN, gΠ = gΠ)

end 

#uncomment the following to check bisection
#test_eqprice = find_eqprice(param)
#@unpack χ, p_star = test 
#collect(χ) 

#======================================================================================================#
# Solving for Equilibrium
#======================================================================================================#

function solve_m(param, solution)
    @unpack F_trans, n_ϕ, gPhi, α, G_prob, D_bar = param
    @unpack χ, gN, p_star = solution

    # New Transition matrix: Transition probability matrix * exit vector χ (element wise, since we do not care about productivities of the firms that exit)
    F_hat = ((1 .- χ).*F_trans)'

    # Invariant distribution is just a homogeneous function of M
    inv_dist_func(M) = M*inv(I - F_hat)*G_prob # this is a function

    # Aggregate demand 
    agg_demand = D_bar/p_star

    # Aggregate supply
    y = @. gPhi*gN^α                                                      #y=zn^α , then a vector of production => y = gPhi*gN^α (gPhi is n_ϕ*1)
    agg_supply = sum(inv_dist_func(1).*y)                                      

    # find mass of entrants (exploit linearity of the invariant distribution)
    M = agg_demand/agg_supply 
	Φ = inv_dist_func(M)

    return M, Φ, F_hat
end

#======================================================================================================#
# Model Statistics
#======================================================================================================#

function ModelStats(param, sol_price, M, Φ)
    @unpack F_trans, n_ϕ, gPhi, α, G_prob, D_bar = param
	@unpack gN, p_star, χ, gΠ = sol_price
	
	# productivity distribution
	pdf_dist = Φ./sum(Φ)
	cdf_dist = cumsum(pdf_dist)
	
	# employment distribution
	emp_dist = Φ.*gN
	pdf_emp = emp_dist./sum(emp_dist)
	cdf_emp = cumsum(pdf_emp)
	
	# exit productivity
	cut_index = findfirst(χ .== 0)
	phicut = param.gPhi[cut_index]

	# stats
	avg_firm_size = sum(emp_dist)/sum(Φ)
	exit_rate = M/sum(Φ)
	Y = sum((gPhi.*gN.^α).*Φ) ## agg production
	emp_prod = sum(emp_dist) # employment used in production
	Pi =  sum(gΠ.*Φ)   # aggregate profits
	
	# employment share
	#size_array = [10, 20, 50, 100, 500]
	
	return (pdf_dist, cdf_dist, pdf_emp, cdf_emp, avg_firm_size, exit_rate, Y, emp_prod, phicut, Pi)
	
end

#======================================================================================================#
# Equilibrium solution
#======================================================================================================#

function SolveModel(param)
	
	# Solve For Prices
	sol_price = find_eqprice(param)  
	M, Φ, F_hat = solve_m(param, sol_price)
    @unpack gN, p_star, χ, gΠ = sol_price
	
	if M<=0;
		println("Warning: No entry, eq. not found.")
	end 
	
	stats = ModelStats(param, sol_price, M, Φ)
    println("Positive entry, M = ", M)
    println("Equilibrium price, p = ", p_star)
	return (sol_price, M, Φ, F_hat, stats)
end

#======================================================================================================#
# Results
#======================================================================================================#

# Call the SolveModel function
solution_info = SolveModel(param)

end   #end for time check 

# Unpack the results
sol_price, M, Φ, F_hat, stats = solution_info;
@unpack p_star, V, χ, gN, gΠ = sol_price;
@unpack n_ϕ, gPhi, F_trans, G_prob = param;
expected_V_incumb = F_trans*V;

## Plots using standard Julia Plot package 
# First plot: Stationary Distribution
p1 = plot(gN, stats[1], label="Firm Share", color=:red, linewidth=2,
          xlabel="size, \$n^*\$", ylabel="Shares",
          xlims=(0, 2000), ylims=(0, 0.05),
          legend=:topright)

plot!(gN, stats[3], label="Emp. Share", color=:green, linewidth=2)

# Second plot: Value Function
p2 = plot(gPhi, V, label="V(x_i; p^*)", linewidth=3, color=:blue,
          xlabel="productivity, \$x_i\$", ylabel="value function, \$V_i(x_i; p^*)\$",
          xlims=(0, 7), ylims=(-25, 100),
          legend=:topright)

plot!(gPhi, expected_V_incumb, label="Expected V", linewidth=3, color=:red)
vline!([stats[9]], label="Cutoff", linestyle=:dash, color=:black)
hline!([0], label="V = 0", linestyle=:dash, color=:black)
hline!([-20], label="V = -c_f", linestyle=:dash, color=:black)

# Combine the plots
combined_plot = plot(p1, p2, layout=(2, 1), size=(800, 800))

# Display the combined plot
display(combined_plot)

#TikZpicture plot for stationary distributions (requires Julia to find LaTeX installation)
#=
@pgf dists = Axis({
    xlabel = "size, \$n^*\$",
    ylabel = "Shares",
    xmin=0, xmax=2000,
    ymin=0, ymax=0.05,
    width = "14cm", height = "11cm",
    legend_pos = "north west",
},
    PlotInc({
        no_marks,
        thick,
        color = "red",
    }, Coordinates(gN, collect(stats[1]))),
    PlotInc({
        no_marks,
        thick,
        color = "green",
    }, Coordinates(gN, collect(stats[3]))),
    Legend(["Firm Share", "Emp. Share"])
)

#Display and save plot
#display(dists)
#pgfsave("/Users/XX/Desktop/plot_dists.pdf", dists)
#pgfsave("/Users/XX/Desktop/plot_dists.tex", dists)

# TikZpicture plot for value function (requires Julia to find LaTeX installation)
@pgf p = Axis({
    xlabel = "productivity, \$x_i\$",
    ylabel = "value function, \$V_i(x_i; p^*)\$",
    ymin = -25, ymax = 100,
    xmin = 0,  xmax = 7, 
    width = "14cm",
    height = "11cm",
    enlarge_x_limits = true,
},
    # Plot the Value Function 
    Plot({
        no_marks,
        ultra_thick,
        color = "blue",  
    }, Table(gPhi, V)),

    # Add the weighted V plot (expected continuation value of incumbents )
    Plot({
        no_marks,
        ultra_thick,
        color = "red",  
    }, Table(gPhi, expected_V_incumb)),
    
    # Add a vertical line for the cutoff
    Plot({
        style = "{dashed, black}",
        no_marks,  # Ensure no markers are used
    }, Coordinates([(stats[9], -25), (stats[9], 100)])),
    
    # Adjust the horizontal line at V=0 to span from xmin to xmax
    Plot({
        style = "{dashed, black}",
    }, Coordinates([(-1, 0), (8, 0)])),  

    # Add a horizontal line at V=-20=-c_f
    Plot({
        style = "{dashed, black}",  
    }, Coordinates([(-1, -20), (8, -20)])) 
);

# Display and save plot 
#display(p)
#pgfsave("/Users/XX/Desktop/plot_gPhi_V.pdf", p)
#pgfsave("/Users/XX/Desktop/plot_gPhi_V.tex", p)

=#