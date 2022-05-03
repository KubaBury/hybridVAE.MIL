using Plots
using Statistics
using Distributions
using LaTeXStrings
using Flux
using Random
using LinearAlgebra
using MLJ
using CategoricalArrays
using EvalMetrics

## Generate data
Random.seed!(12394)
p=100   #number of observations in class
X, y = make_moons(2*p; noise=0.24,yshift=0.5)
x = hcat(X.x1, X.x2)' #reimplement
y_oh = Flux.onehotbatch((levelcode.(y))[:],1:2) #onehot

#visualize
scatter(x[1,y.==0], x[2,y.==0], label="y=0, true", legendfontsize=10,yguidefontsize=20, xguidefontsize=20)
scatter!(x[1,y.==1],x[2,y.==1], label="y=1, true")


## NN == inicializace
nx = 2 
nz = 2 
nh = 20 

q_yz =  Chain(Dense(nx+2,nh+5,selu),Dense(nh+5,nh)) # encoder
μ, logσ = Dense(nh, nz,selu), Dense(nh, nz,selu)
predictor = Chain(Dense(nx,nh, selu),Dense(nh,2),softmax) #predictor
f = Chain(Dense(nz+2,nh,selu),Dense(nh,nx)) #decoder

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

z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ)) #samples
KL(μ, logσ) = 0.5 *sum(-1.0 .-  2 .*logσ .+ (exp.(logσ)).^2 .+ μ.^2, dims=1) #KLdiv

s=0.13 #weight for KL div
function L(x,y_oh)      #VAE loss          
    HID = encoder(x,y_oh)
    zsample = z(μ(HID),logσ(HID))
    K = 0.5*diag((x.-decoder(zsample,y_oh))' *(x.-decoder(zsample,y_oh))) .+ s*KL(μ(HID),logσ(HID))
    return K
end

function L2(x,y_oh)
    a = Flux.logitcrossentropy(predictor(x), y_oh) #classification loss
    return a
end

function hybridVAE(x,y_oh)  
    yp = predictor(x)
    y_oh1 = Flux.onehotbatch((ones(2*p).+1)[:],1:2)
    y_oh2 = Flux.onehotbatch((zeros(2*p).+1)[:],1:2)
    Li = yp[1,:].*L(x,y_oh1) + yp[2,:].*L(x,y_oh2) 
    LL = sum(Li) - Flux.logitcrossentropy(yp,yp,agg=sum)
   return LL
end

function hybridloss1(x,y_oh) #total loss for separate model
    sum(L(x,y_oh)) + L2(x,y_oh)
end

β=10500
function hybridloss2(x,y_oh) #total loss for hybrid model
    hybridVAE(x,y_oh) + β*L2(x,y_oh)
end

#train
ps = Flux.params(q_yz, μ, logσ, f,predictor)
da = Iterators.repeated((x,y_oh),10000)
opt = Flux.ADAM()
Flux.train!(hybridloss2,ps,da,opt); 


#test
c=100
Random.seed!(12395)
Xtest, ytest = make_moons(c; noise=0.24,yshift=0.5)
x_test = hcat(Xtest.x1, Xtest.x2)'
predictor(x_test)
Zs=randn(nz,c)
y_test = levelcode.(ytest)
yoh_test = Flux.onehotbatch((y_test)[:],1:2)
Xg=decoder(Zs,yoh_test) 
q = 0.5*diag((x.-decoder(zsample,y_oh))' *(x.-decoder(zsample,y_oh)))

#visualize
scatter!(Xg[1,y_test.==1],Xg[2,y_test.==1], label="y=0, estimated",marker=:cross,markersize=6)
scatter!(Xg[1,y_test.==2],Xg[2,y_test.==2], label="y=1, estimated",marker=:cross, markersize=6,legendfontsize=11,
yguidefontsize=15, xguidefontsize=15,xtickfontsize=15,
ytickfontsize=15,)

#Q=[500, 1000,2000,4000,6000,8000,9000,9500,10000,10500,11000,12000,15000,20000,40000,80000,160000]


#function betaiter(betas)
 #   S = zeros(length(betas))
  #  for i = 1:length(betas)
   #     S[i] = beta(betas[i])
   # end
    #return S
#end    

#A = betaiter(Q)
#
#plot(Q[1:16],A[1:16],legendfontsize=11,
#yguidefontsize=15, xguidefontsize=15,xtickfontsize=14,
#ytickfontsize=14,linewidth=3,legend=:bottomright,xlims=(-100,80000),
#xticks=[0, 10000, 35000, 60000], label="AUC_HM",xlabel=L"\nu",ylabel=L"AUC" )
#plot!([0,80000],[0.954,0.954],lw=3,linestyle=:dash,label="AUC_SM")