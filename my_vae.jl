using Plots
using Statistics
using Distributions
using LaTeXStrings
using Flux
using Random
using LinearAlgebra
## Generate data
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
Random.seed!(12394)

p=50
B=rand(MvNormal([2,2], [3 2; 2 3]), p)
x1=B[1,:]
x2=B[2,:]
x=[x1 x2]'


## NN == inicializace??
nx = 2
nz = 2
nh = 40

A, μ, logσ = Dense(nx, nh), Dense(nh, nz,selu), Dense(nh, nz,selu)

function encoder(x)
    h = A(x)
    return h
end

f = Chain(Dense(nz,nh,selu),Dense(nh,nx)) 
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)


s = 0.2 #weight for KL
function loss(x)
    HID = encoder(x)
    zsample = z(μ(HID),logσ(HID))
    L = 0.5*sum((x.-f(zsample)).^2) .+ s*KL(μ(HID),logσ(HID)) 
    return L
end

#train
ps = Flux.params(A, μ, logσ, f)
da = Iterators.repeated((x,),10000)
opt = Flux.ADAM()
Flux.train!(loss,ps,da,opt);
#println(loss([x y]'))
#test
Zs=randn(nz,p)
Xg=f(Zs) 
#Zg=g([x y]')
#scatter(x,y, label = "true", markersize = 8, legend =:topleft, xlabel = "x", ylabel = "y", markershape=:cross, c=:red)
#scatter!(Xg[1,:], Xg[2,:], label = "estimated", markersize = 8, markershape=:cross,c=:green)


#plt(x1,x2) = pdf(MvNormal([2,2], [3 2; 2 3]), [x1,x2])
#plt2(x1,x2) = pdf.(MvNormal(mean(f(z(μ(encoder(x[:,1])),exp.(logσ(encoder(x[:,1]))))), dims=2)[:],[1 0; 0 1]), [x1,x2])


#visualize
gr()
#c1=contour(-2.:0.01:6.2, -2.:0.01:6.2, (x1,x2)->plt(x1,x2), linewidth=1, legendfontsize=30,yguidefontsize=20, xguidefontsize=20,
#xtickfontsize=20,ytickfontsize=20,levels=6,alpha=0.8,c =:blues,colorbar=false,lw=5)
scatter(x[1,:], x[2,:],label = "true", markersize = 8, markershape=:cross,c=:green,legend=:bottomright
,legendfontsize=15,yguidefontsize=14, xguidefontsize=14, 
 xtickfontsize=20,ytickfontsize=20)
scatter!(Xg[1,:], Xg[2,:],label = "estimated", markersize = 8, 
markershape=:cross,c=:red,legend=:bottomright, legendfontsize=15,yguidefontsize=20, xguidefontsize=20)

#contour!(c1,-2:0.1:6.2, -2.:0.1:6.2, (x1,x2) ->plt2(x1,x2), linewidth=5, levels=6,linestyle=:dash,alpha=1,lw=5, c=:reds)



#scatter!(Zg[2][1,:],Zg[1][1,:])
#k = histogram(Zg[2]', nbins = 50, normalize=:pdf, color =:lightgreen, label = false)
#c = -3:0.01:3
#ng = 1/(2*pi)^(1/2)*exp.(-c.^2/2)
#l = plot!(k ,c,ng, linewidth = 5, color =:brown, xlabel = "x", ylabel = "f(x)", label = L"N(0,1)")
 