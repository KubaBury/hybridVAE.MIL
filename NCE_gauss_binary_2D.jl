using Distributions
using Flux
using Flux: @epochs, throttle, @show, train!
using Plots
using Random
using LinearAlgebra
using LaTeXStrings

Random.seed!(1214)#seed0
N = 100; # number of samples
dataDist = MvNormal([5,2],[2 1;1 2]); #distribution of the data
x = rand(dataDist, N) #training data
noiseDist = MvNormal([2, 2], [10 0;0 10]); #distribution of the noise data
y = rand(noiseDist, N) #noise data


"""
Computes log-likelihood for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function Gaussianlogpdf(u, μ, A, c)
    inside = .-0.5.*(u .-μ)'*inv(A*A')*(u .-μ).+c
    l = diag(inside)
    return l
end

"""
Computes function G for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function G(u, μ, A, c)
    G = Gaussianlogpdf(u, μ, A, c) .- logpdf(noiseDist, u)
    return G
end

"""
Computes function h for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function h(u, μ::Vector{Float64}, A, c)
    h = 1 ./ (1 .+ exp.(-G(u, μ, A, c)))
    return h
end

"""
Computes loss function for given vector of data x and noise data y.
"""

function loss_binary(x, y)
    loss = -sum(log.(h(x, μ, A, c)) .+ log.(1 .- h(y, μ, A, c)))
    return loss
end


μ = [1.0, 1.0]
A = [1.0 0.0;0.0 1.0]
c = [1.0]



# set initial value for training

L = Float64[]
opt = Flux.ADAM() #optimalization method
evalcb() = push!(L, loss_binary(x, y))
Flux.train!((x, y) -> loss_binary(x, y), [μ, A, c], Iterators.repeated((x, y), 10000), opt, cb = throttle(evalcb, 1)) # train binary
println("μ=$μ, Σ=$(A*A'), c=$c") #results

plotlyjs()
plot(1:length(L), L, xlabel = "time[s]", ylabel = "loss",
linewidth = 5, legend=:false ,
legendfontsize=30,yguidefontsize=20, xguidefontsize=20,
xtickfontsize=20,ytickfontsize=20)

#plc=contour(0.3:0.01:2.2,0:0.01:2,(m,s)->joint(1,2,m,s), xlabel="m", ylabel="s", levels=6, linewidth = 2);
plt(x1,x2) = pdf(MvNormal([5, 2],[2 1;1 2]), [x1,x2])
plt2(x1,x2) = pdf(MvNormal(μ,A*A'), [x1,x2])

gr()
c1=contour(1.:0.01:9.2, -2.:0.01:6.2, (x1,x2)->plt2(x1,x2), linewidth=1, legendfontsize=30,yguidefontsize=20, xguidefontsize=20,
xtickfontsize=20,ytickfontsize=20,levels=6,alpha=0.8,c =:blues,colorbar=false,lw=5)
contour!(c1,1.7:0.01:8.2, -2.:0.01:6.2, (x1,x2) ->plt(x1,x2), 
linewidth=5, levels=6,linestyle=:dash,alpha=1,lw=5, c=:reds)


