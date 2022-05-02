using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated

problem  = "Musk1"

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

data = "D:/VU/SCRIPTS/DataSets/Musk1"
(x,y) = csv2mill(data)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)

# create the model
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.tanh)),                      # model on the level of Flows
    meanmax_aggregation(10),                                       # aggregation
    ArrayModel(Chain(Dense(21, 10, Flux.tanh), Dense(10, 2))))  # model on the level of bags

# define loss function
loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh) 

# the usual way of training
evalcb = throttle(() -> @show(loss(x, y_oh)), 1)
opt = Flux.ADAM()
@epochs 10 Flux.train!(loss, Flux.params(model), repeated((x, y_oh), 1000), opt, cb=evalcb)

# calculate the error on the training set (no testing set right now)
1-mean(mapslices(argmax, model(x).data, dims=1)' .!= y.+1)