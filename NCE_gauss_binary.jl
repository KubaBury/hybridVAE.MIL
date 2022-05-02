using Distributions
using Flux
using Flux: @epochs, throttle, @show, train!
using Plots
using Random
using LaTeXStrings

Random.seed!(12314)#seed
N = 100; # number of samples
dataDist = Normal(5,2); #distribution of the data
x = rand(dataDist, N) #training data
noiseDist = Normal(0, 10); #distribution of the noise data
y = rand(noiseDist, N) #noise data


"""
Computes log-likelihood for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function Gaussianlogpdf(u::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, c::Vector{Float64})
    if σ >= [0.0]
        l = .-0.5 .* ((u .- μ) ./ σ) .^ 2 .+ c #log likelihood of the Gaussian      
        return l
    else
        println("σ should be greater than 0")
    end
end

"""
Computes function G for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function G(u::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, c::Vector{Float64})
    if σ >= [0.0]
        G = Gaussianlogpdf(u, μ, σ, c) .- logpdf.(noiseDist, u)
        return G
    else
        println("σ should be greater than 0")
    end
end

"""
Computes function h for given vector of data u with parameters mean μ, 
variance σ and normalization constant c. Note that σ should be greater than 0.
"""
function h(u::Vector{Float64}, μ::Vector{Float64}, σ::Vector{Float64}, c::Vector{Float64})
    h = 1 ./ (1 .+ exp.(-G(u, μ, σ, c)))
    return h
end

"""
Computes loss function for given vector of data x and noise data y.
"""
function loss_binary(x::Vector{Float64}, y::Vector{Float64})
    loss = -sum(log.(h(x, μ, σ, c)) .+ log.(1 .- h(y, μ, σ, c)))
    return loss
end
μ = [0.0]
σ = [1.0]
c = [1.0]


# set initial value for training

L = Float64[]
opt = Flux.ADAM() #optimalization method
evalcb() = push!(L, loss_binary(x, y))
Flux.train!((x, y) -> loss_binary(x, y), [μ, σ, c], Iterators.repeated((x, y), 10000), opt, cb = throttle(evalcb, 1)) # train binary
println("μ=$μ, σ=$σ, c=$c") #results



#plotlyjs() #initalize backend
plotlyjs()
plot(1:length(L), L, xlabel = "time[s]", ylabel = "loss",
linewidth = 5, legend=:false ,
legendfontsize=30,yguidefontsize=20, xguidefontsize=20,
xtickfontsize=20,ytickfontsize=20)
gr()
B = plot(-2:0.01:12, pdf.(Normal(5.0, 2.0), -2:0.01:12),
linewidth = 5, fill = (0, :lightblue), fillalpha = 0.3, legend=:false,
legendfontsize=16,yguidefontsize=20, xguidefontsize=20,
xtickfontsize=20,ytickfontsize=20)

plot!(-2:0.01:12, exp.(Gaussianlogpdf(collect(-2:0.01:12), μ, σ, c)),
linewidth = 5, fill = (0, :red), fillalpha = 0.3,
legendfontsize=16,yguidefontsize=20, xguidefontsize=20,
xtickfontsize=20,ytickfontsize=20)