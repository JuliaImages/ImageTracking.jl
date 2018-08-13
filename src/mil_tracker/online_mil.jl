mutable struct Stump{I <: Int, F <: Float64, B <: Bool}
    #Initialized
    index::I
    μ0::F
    μ1::F
    σ0::F
    σ1::F
    l_rate::F
    trained::B

    #Uninitialized
    q::F
    s::I
    log_n0::F
    log_n1::F
    e0::F
    e1::F

    Stump(index::I = -1, μ0::F = 0.0, μ1::F = 0.0, σ0::F = 1.0, σ1::F = 1.0, l_rate::F = 0.85, trained::B = false) where {I <: Int, F <: Float64, B <: Bool} = new{I, F, B}(
          index, μ0, μ1, σ0, σ1, l_rate, trained)
end

mutable struct MIL_Boost_Classifier{I <: Int, F <: Float64}
    num_sel::I
    num_feat::I
    l_rate::F
    num_samples::I
    counter::I
    weak_clf::MVector{N, Stump} where N
    selectors::Set{I}

    MIL_Boost_Classifier(num_sel::I, num_feat::I, l_rate::F, num_samples::I = 0, counter::I = 0) where {I <: Int, F <: Float64} = new{I, F}(
                         num_sel, num_feat, l_rate, num_samples, counter)
end

#Misc

@inline function sigmoid(x::Real)
    1/(1 + exp(-x))
end

#Stump

#TODO: Check the last two parameters
function update(stump::Stump, posx::Array{T, 2}, negx::Array{T, 2}, posw::Array{Float64, 2}, negw::Array{Float64, 2}) where T
    if size(posx)[2] > 0
        pos_μ = mean(posx[:, stump.index])
    else
        pos_μ = 0.0
    end
    if size(negx)[2] > 0
        neg_μ = mean(negx[:, stump.index])
    else
        neg_μ = 0.0
    end

    if stump.trained
        if size(posx)[2] > 0
            stump.μ1 = (1 - stump.l_rate)*pos_μ + stump.μ1*stump.l_rate
            diff = posx[:,stump.index] .- μ1
            stump.σ1 = (1 - stump.l_rate)*mean(diff.*diff) + stump.σ1*stump.l_rate
        end
        if size(negx)[2] > 0
            stump.μ0 = (1 - stump.l_rate)*neg_μ + stump.μ0*stump.l_rate
            diff = negx[:,stump.index] .- μ0
            stump.σ0 = (1 - stump.l_rate)*mean(diff.*diff) + stump.σ0*stump.l_rate
        end
    else
        stump.trained = true
        if size(posx)[2] > 0
            stump.μ1 = pos_μ
            σ = std(posx[:,stump.index])
            stump.σ1 = σ*σ + 10^-9
        end
        if size(negx)[2] > 0
            stump.μ0 = neg_μ
            σ = std(negx[:,stump.index])
            stump.σ0 = σ*σ + 10^-9
        end
    end

    stump.q = (stump.μ1 - stump.μ0)/2
    stump.s = sign(stump.μ1 - stump.μ0)
    stump.log_n0 = log(1/(sqrt(stump.σ0)))
    stump.log_n1 = log(1/(sqrt(stump.σ1)))
    stump.e0 = -1/(2*stump.σ0)
    stump.e1 = -1/(2*stump.σ1)
end

function classify(stump::Stump, x::Array{T, 2}, i::Int) where T
    xx = x[stump.index, i]
    log_p0 = stump.e0*(xx - stump.μ0)^2 + stump.log_n0
    log_p1 = stump.e1*(xx - stump.μ1)^2 + stump.log_n1
    return log_p1 > log_p0
end

function classifyF(stump::Stump, x::Array{T, 2}, i::Int) where T
    xx = x[stump.index, i]
    log_p0 = stump.e0*(xx - stump.μ0)^2 + stump.log_n0
    log_p1 = stump.e1*(xx - stump.μ1)^2 + stump.log_n1
    return log_p1 - log_p0
end

function classifySetF(stump::Stump, x::Array{T, 2}) where T
    res = SharedVector{Float64}(size(x)[1])
    @parallel for i = 1:length(res)
        res[i] = classifyF(stump, x, i)
    end
    return res
end

#MIL_Boost_Classifier

function initialize_ClfMilBoost(num_sel::Int = 50, num_feat::Int = 250, l_rate::Float64 = 0.85)
    boost_classifier = MIL_Boost_Classifier(num_sel, num_feat, l_rate)
    weak_clf = Vector{Stump}()
    for i = 1:num_feat
        push!(weak_clf, Stump(i))
        weak_clf[i].l_rate = l_rate
    end
    boost_classifier.weak_clf = MVector{num_feat, Stump}(weak_clf)
    boost_classifier.selectors = Set{Int}()
    return boost_classifier
end

function update(boost_classifier::MIL_Boost_Classifier, posx::Array{T, 2}, negx::Array{T, 2}) where T
    Hpos = SharedVector{Float64}(size(posx)[1])
    Hneg = SharedVector{Float64}(size(negx)[1])

    posw = MVector{size(posx)[1], Float64}()
    negw = MVector{size(negx)[1], Float64}()
    pospred = SharedArray{Float64}(length(boost_classifier.weak_clf), size(posx)[1])
    negpred = SharedArray{Float64}(length(boost_classifier.weak_clf), size(negx)[1])

    @parallel for i = 1:boost_classifier.num_feat
        update(boost_classifier.weak_clf[i], posx, negx)
        pospred[i, :] = classifySetF(boost_classifier.weak_clf[i], posx)
        negpred[i, :] = classifySetF(boost_classifier.weak_clf[i], negx)
    end

    for i = 1:boost_classifier.num_sel
        poslikl = SharedVector{Float64}(length(boost_classifier.weak_clf))
        neglikl = SharedVector{Float64}(length(boost_classifier.weak_clf))
        likl = SharedVector{Float64}(length(boost_classifier.weak_clf))

        @parallel for j = 1:length(boost_classifier.weak_clf)
            lll = 1.0
            for k = 1:size(posx)[1]
                lll *= (1 - sigmoid(Hpos[k] + pospred[j, k]))
            end
            poslikl[j] = -log(1 - lll + 10^-5)

            lll = 0.0
            for k = 1:size(negx)[1]
                lll += -log(1 - sigmoid(Hneg[k] + negpred[j, k]) + 10^-5)
            end
            neglikl[j] = lll

            likl[j] = (poslikl[j]/size(posx)[1]) + (neglikl[j]/size(negx)[1])
        end

        order = sortperm(likl)
        for j = 1:length(order)
            if in(order[j], boost_classifier.selectors) == 0
                push!(boost_classifier.selectors, order[j])
                break
            end
        end

        @parallel for j = 1:size(posx)[1]
            Hpos[j] += pospred[boost_classifier.selectors[i], j]
        end
        @parallel for j = 1:size(negx)[1]
            Hneg[j] += negpred[boost_classifier.selectors[i], j]
        end
    end

    boost_classifier.counter += 1
end

function classify(boost_classifier::MIL_Boost_Classifier, x::Array{T, 2}, log_R::Bool = true) where T
    res = SharedVector{Float64}(size(x)[1])

    for i in boost_classifier.selectors
        tr = classifySetF(boost_classifier.weak_clf[i], x)
        @parallel for j = 1:size(x)[1]
            res[j] += tr[j]
        end
    end

    if !log_R
        res = sigmoid.(res)
    end

    return res
end
