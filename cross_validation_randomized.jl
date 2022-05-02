using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using Random: shuffle
using Plots


###BrownCreeper, CorelAfrican,CorelBeach, Elephant,Fox, Musk1, Musk2, Mutagenesis1(2)
###Newsgroups1, Newsgroups2, Newsgroups3, Protein, Tiger, Web1(2,3,4), WinterWren

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

### load data ###
data = "D:/VU/SCRIPTS/DataSets/Tiger"
(x,y) = csv2mill(data);
y_oh = Flux.onehotbatch((y.+1)[:],1:2);

### cross-validation - specify number of folds and dense layer ###### 
K=100
W=20
h=1
A = length(x.data.data[:,1]) # for dense layer
b = length(y) #number of bags
#function KfoldCV(K,W;h=1)

	n = floor(Int, length(y)*0.2) # number of bags in each fold
	L1 = zeros(W,K) # initial position of loss array
	L2 = zeros(W,K)
	L3 = zeros(W,K)
	L4 = zeros(W,K)
	tr_set = zeros(Int, b-n, K)
	te_set = zeros(Int, n, K)

	opt = Flux.ADAM();
	function hybridloss(ŷ, y, α; agg=mean)
		agg(.-sum( α.*(y .* logsoftmax(ŷ; dims = 1)) + (1-α).*(y .* logsoftmax(ŷ; dims = 2)); dims = 1))   
	end 
	### define random training and testing samples
	for  j =1:K
		r1 = shuffle(1:b)
		r2 = sample(1:b, n, replace = false)
		q = symdiff(r1, r2)
		tr_set[:,j] = q 
		te_set[:,j] = r2
	end
	for i = 1:W
		# create the model
		# model on the level of bags
	
		#loss function
		loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);
		loss1(x,y_oh) = hybridloss(model(x).data, y_oh, 0.5)
		for  m = 1:K
		#train
		model = BagModel(
    	ArrayModel(Dense(A, h*i, Flux.tanh)),                      			# model on the level of Flows
    	meanmax_aggregation(h*i),                                      		# aggregation
		ArrayModel(Chain(Dense(2*h*i+1, h*i, Flux.tanh), Dense(h*i, 2)))) 
		Flux.train!(loss1, Flux.params(model), repeated((x[tr_set[:,m]], y_oh[:,tr_set[:,m]]), 1000), opt);
			# calculate loss for each fold and dense layer
			L1[i,m] = loss(x[te_set[:,m]], y_oh[:,te_set[:,m]])
			L2[i,m] = loss(x[tr_set[:,m]], y_oh[:,tr_set[:,m]])
			
	end
end
	#visualization
	x2 = mean(L1, dims = 2)
	x3 = mean(L2, dims = 2)
	x1 = 1:W
	plot(x1, x2, xlabel = "Model Complexity", ylabel = "Prediction Error", 
	label = L"$\mathrm{HDGM}:\alpha=0.5$",lw=3,legend=:bottomright, color=:green, title="HDGM-0.5 vs. Supervised; Tiger",
	xtickfont=font(11), 
    ytickfont=font(11),
    guidefont=font(14),
    legendfont=font(11))
	plot!(x1, x3, label = "Train data", lw=3, legend=:bottomright, color=:red)
	plot!(x1, D1[1], lw=3, color=:blue, label ="Supervised")
#end

