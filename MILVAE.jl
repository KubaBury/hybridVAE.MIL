using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using EvalMetrics
using Random
using Plots
using Zygote

function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

function csv2mill(problem)
	x=readdlm("$(problem)/data.csv",'\t',Float32)
	bagids = readdlm("$(problem)/bagids.csv",'\t',Int)[:]
	bags = seqids2bags(bagids)
	y = readdlm("$(problem)/labels.csv",'\t',Int)
	y = map(b -> maximum(y[b]), bags)
	(samples = BagNode(ArrayNode(x), bags), labels = y)
end
function labels2instances(yb,xbl)
    ym=map(x->x[1]*ones(x[2]),zip(yb,xbl))
	y=vcat(ym...)
	return [y';(1.0 .-y)']
end 
function labels2instances(x)
    xbl = Zygote.@ignore (length.(x.bags))
    yb = softmax(model(x))[1,:]
    labels2instances(yb,xbl)
end

function split_train_test(x,y,ratio)
	A = length(x.data.data[:,1]) 
	b = length(y) 
	n = floor(Int, length(y)*(1-ratio)) 
	tr_set = zeros(Int, b-n)
	te_set = zeros(Int, n)
	r1 = shuffle(1:b)
	r2 = sample(1:b, n, replace = false)
	q = symdiff(r1, r2)
	tr_set[:] = q 
	te_set[:] = r2
	x_train = x[tr_set]
	x_test = x[te_set]
	y_train = y[tr_set]
	y_test = y[te_set]
	(x_train,y_train), (x_test, y_test)
end


data = "D:/VU/SCRIPTS/DataSets/Musk1"
(x,y) = csv2mill(data)
y_oh= Flux.onehotbatch((y.+1)[:],1:2) 
#y_oh_i = labels2instances(x)  #uvnitr se vola model!!! 
y_oh_i = labels2instances(y,length.(x.bags))  

(x, y),(x_test, y_test) = split_train_test(x,y,0.8)
y_oh_test = Flux.onehotbatch((y_test.+1)[:],1:2)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)

v=size(x.data.data)[2]

model = BagModel(
    Dense(size(x.data.data)[1], 10, Flux.tanh),
    BagCount(SegmentedMeanMax(10)),
    Chain(Dense(21, 20, Flux.tanh),Dense(20,10,Flux.tanh), Dense(10, 2)))

# define loss function
loss(x, y_oh) = Flux.logitcrossentropy(model(x), y_oh) 

## NN == inicializace
nx = size(x.data.data)[1]
nz = 16 #(4 8 16 32 64)
nh = 32 #prizpusobit

q_yz =  Chain(Dense(nx+2,nh+5,tanh),Dense(nh+5,nh)) # encoder
μ, logσ = Dense(nh, nz,tanh), Dense(nh, nz,tanh)
f = Chain(Dense(nz+2,nh,tanh),Dense(nh,nx)) #decoder

function encoder(x,y_oh_i)
    v=vcat(x.data.data,y_oh_i)
    h = q_yz(v)
    return h
end

function decoder(z,y_oh_i)
    v=vcat(z,y_oh_i)
    h = f(v)
    return h
end

z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ)) #samples
KL(μ, logσ) = 0.5 *sum(-1.0 .-  2 .*logσ .+ (exp.(logσ)).^2 .+ μ.^2, dims=1) #KLdiv



s=0.02 #(0.02 pro hybrid loss1, 0.08 pro hybrodloss2)
function L(x,y_oh_i)      #VAE loss    , x jsou instance      
    HID = encoder(x,y_oh_i)
    zsample = z(μ(HID),logσ(HID))
    # VS!! diag?
    K = 0.5*sum((x.data.data .-decoder(zsample,y_oh_i)).^2) + s*sum(KL(μ(HID),logσ(HID)))
    return K
end

function hybridVAE(x)   
    yp = labels2instances(x)
    y_oh1 = Flux.onehotbatch((ones(v).+1)[:],1:2)
    y_oh2 = Flux.onehotbatch((zeros(v).+1)[:],1:2)
    Li = yp[1,:].*L(x,y_oh1) + yp[2,:].*L(x,y_oh2) 
    LL = sum(Li) - Flux.logitcrossentropy(yp,yp,agg=sum)
   return LL
end

β = 10000000000
function hybridloss2(x,y_oh)
    hybridVAE(x) + β*loss(x,y_oh)
end

ps = Flux.params(q_yz, μ, logσ, f,model)
da = Iterators.repeated((x,y_oh),10000)
opt = Flux.ADAM(0.001)
Flux.train!(hybridloss2,ps,da,opt)


rocplot(y_test,softmax(model(x_test))[2,:])
au_roccurve(y_test,softmax(model(x_test))[2,:])

