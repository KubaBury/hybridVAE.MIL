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

p=100
B=rand(MvNormal([0,0],[1 0 ; 0 1]),p)
C=rand(MvNormal([5,5],[1 0 ; 0 1]),p)
x = hcat(B,C)
y1 = ones(p)
y2 = zeros(p)
y = vcat(y1,y2)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)

## NN == inicializace??
nx = 2
nz = 2
nh = 20


q_yz, μ, logσ = Dense(nx+2, nh), Dense(nh, nz,selu), Dense(nh, nz,selu)
predictor = Chain(Dense(nx,nh),Dense(nh,2),softmax)
f = Chain(Dense(nz+2,nh,swish),Dense(nh,nx)) 


function encoder(x,y_oh)
    v=vcat(x,y_oh)
    h = q_yz(v)
    return h
end

function decoder(z,y_oh)
    v=vcat(z,y_oh)
    h = f(v)
    return h
end

z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum(-1.0 .-  2 .*logσ .+ (exp.(logσ)).^2 .+ μ.^2)

s=0.5
function L(x,y_oh)
    HID = encoder(x,y_oh)
    zsample = z(μ(HID),logσ(HID))
    K = 0.5*sum((x.-decoder(zsample,y_oh)).^2) .+ s*KL(μ(HID),logσ(HID))
    return K
end

#function hybridVAE(x)
 #   yp = predictor(x)
  #  y_oh1 = Flux.onehotbatch((ones(2*p).+1)[:],1:2)
  #  y_oh2 = Flux.onehotbatch((zeros(2*p).+1)[:],1:2)
  #  Li = yp[1,:].*L(x,y_oh1) + yp[2,:].*L(x,y_oh2)
   # LL = sum(Li)
   # return LL
#end


L2(x,y_oh)=Flux.logitcrossentropy(predictor(x), y_oh)

α = 0.5
function hybridloss(x,y_oh)
    (1-α)*L(x,y_oh) + α*L2(x,y_oh)
end

ps = Flux.params(q_yz, μ, logσ, f,predictor)
da = Iterators.repeated((x,y_oh),20000)
opt = Flux.ADAM(0.01)
Flux.train!(hybridloss,ps,da,opt);
#println(loss([x y]'))

Q=50
x_test1=rand(MvNormal([5, 5],[1.1 0 ;0 1.1]),Q)
x_test2=rand(MvNormal([0, 0],[1.1 0 ;0 1.1]),Q)
x_test=hcat(x_test1,x_test2)
predictor(x_test)
Zs=randn(nz,2*Q)
y_test1 = zeros(Q)
y_test2 = ones(Q)
y_test = vcat(y_test1,y_test2)
yoh_test = Flux.onehotbatch((y_test.+1)[:],1:2)
Xg=decoder(Zs,yoh_test) 
scatter(x[1,1:p],x[2,1:p])
scatter!(x[1,p+1:2*p],x[2,p+1:2*p])
scatter!(Xg[1,1:Q],Xg[2,1:Q])
scatter!(Xg[1,Q+1:2*Q],Xg[2,Q+1:2*Q])


#Zg=g([x y]')
#scatter(x,y, label = "true", markersize = 8, legend =:topleft, xlabel = "x", ylabel = "y", markershape=:cross, c=:red)
#scatter!(Xg[1,:], Xg[2,:], label = "estimated", markersize = 8, markershape=:cross,c=:green)


#plt(x1,x2) = pdf(MvNormal([2,2], [3 2; 2 3]), [x1,x2])
#plt2(x1,x2) = pdf.(MvNormal(mean(f(z(μ(encoder(x[:,1])),exp.(logσ(encoder(x[:,1]))))), dims=2)[:],[1 0; 0 1]), [x1,x2])

#gr()
#c1=contour(-2.:0.01:6.2, -2.:0.01:6.2, (x1,x2)->plt(x1,x2), linewidth=1, legendfontsize=30,yguidefontsize=20, xguidefontsize=20,
#xtickfontsize=20,ytickfontsize=20,levels=6,alpha=0.8,c =:blues,colorbar=false,lw=5)
#scatter(x,y,label = "true", markersize = 8, markershape=:cross,c=:green,legend=:bottomright
#,legendfontsize=15,yguidefontsize=14, xguidefontsize=14, 
#xlabel="x", ylabel="y", xtickfontsize=20,ytickfontsize=20)
#scatter!(Xg[1,:], Xg[2,:],label = "estimated", markersize = 8, 
#markershape=:cross,c=:red,legend=:bottomright, legendfontsize=15,yguidefontsize=20, xguidefontsize=20)

#contour!(c1,-2:0.1:6.2, -2.:0.1:6.2, (x1,x2) ->plt2(x1,x2), linewidth=5, levels=6,linestyle=:dash,alpha=1,lw=5, c=:reds)



#scatter!(Zg[2][1,:],Zg[1][1,:])
#k = histogram(Zg[2]', nbins = 50, normalize=:pdf, color =:lightgreen, label = false)
#c = -3:0.01:3
#ng = 1/(2*pi)^(1/2)*exp.(-c.^2/2)
#l = plot!(k ,c,ng, linewidth = 5, color =:brown, xlabel = "x", ylabel = "f(x)", label = L"N(0,1)")
 