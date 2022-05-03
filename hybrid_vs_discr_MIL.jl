using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using EvalMetrics
using Random
using Plots
using LaTeXStrings

#loading data fucntions
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
#define hybrid loss
function logithybridloss(ŷ, y, α; agg=mean)
	agg(.-sum( α.*(y .* logsoftmax(ŷ; dims = 1)) + (1-α).*(y .* logsoftmax(ŷ; dims = 2)); dims = 1))   
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

#split test train
data = "D:/VU/SCRIPTS/DataSets/Fox"
(x,y) = csv2mill(data)
(x_train, y_train),(x_test, y_test) = split_train_test(x,y,0.8)
y_oh_test = Flux.onehotbatch((y_test.+1)[:],1:2)
y_oh_train = Flux.onehotbatch((y_train.+1)[:],1:2)





function auc()
# define model
model = BagModel(
    Dense(size(x.data.data)[1], 10, Flux.tanh),
    BagCount(SegmentedMeanMax(10)),
    Chain(Dense(21, 10, Flux.tanh), Dense(10, 2)))


loss(x, y_oh) = Flux.logitcrossentropy(model(x), y_oh) #cross entropy loss
loss2(x,y_oh) = logithybridloss(model(x),y_oh,0.5) #hybrid loss
# the usual way of training
opt = Flux.ADAM()
Flux.train!(loss, Flux.params(model), repeated((x_train, y_oh_train), 1000), opt)

# calculate the error on the training set (no testing set right now)
a = au_roccurve(y_test,softmax(model(x_test))[2,:])
return a
end

 #iterate over more runs
function iter(pocet)
	A=zeros(pocet)
	for i in 1:pocet
		A[i] = auc()
	end
	return mean(A),std(A)
end