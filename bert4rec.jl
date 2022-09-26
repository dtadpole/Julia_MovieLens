include("./data.jl")
include("./util.jl")

using Distributions
using Transformers
using Transformers.Basic
using ProgressMeter: Progress, next!, finish!
using Zygote: pullback
using OneHotArrays
using Serialization
using JLD

# load ratings
ratings = load_ratings()
USER_SIZE = max(ratings.uid...)
MOVIE_SIZE = max(ratings.mid...)

# load users
# users = load_users()

# load movies
# movies = load_movies()


# special token for MASK and NULL
NULL_VALUE = MOVIE_SIZE + 1
MASK_VALUE = MOVIE_SIZE + 2

# prep user rating sequences
USER_RATING_SEQUENCES = fill(Vector{Int}(), USER_SIZE)

for u in 1:USER_SIZE
    USER_RATING_SEQUENCES[u] = ratings[ratings.uid.==u, :mid]
end

HOLDOUT_SIZE = 600

TRAIN_SEQUENCES = USER_RATING_SEQUENCES[1:end-HOLDOUT_SIZE*2]
VALIDATE_SEQUENCES = USER_RATING_SEQUENCES[end-HOLDOUT_SIZE*2+1:end-HOLDOUT_SIZE]
TEST_SEQUENCES = USER_RATING_SEQUENCES[end-HOLDOUT_SIZE+1:end]

DIM = args["model_dim"]

MASK_RATIO = args["mask_ratio"]

# custom attention layer (for future experiments)
struct CustomAttention
    head::Int
    dim::Int
    input::Dense
    output::Dense
    dropout::Float32
end

# constructor
function CustomAttention(head, dim; dropout=0.1f0)
    if dim % head != 0
        error("Dimension must be divisible by head")
    end
    CustomAttention(head, dim, Dense(dim => dim * 3), Dense(dim => dim), dropout)
end

Flux.@functor CustomAttention

(m::CustomAttention)(x::AbstractArray) = begin

    # println(size(x))
    seq_len = size(x, 2)

    x = m.input(x)                                                      # (DIM, SEQ_LEN, BATCH_SIZE) => (DIM * 3, SEQ_LEN, BATCH_SIZE)

    k, q, v = x[1:m.dim, :, :], x[m.dim+1:m.dim*2, :, :], x[m.dim*2+1:m.dim*3, :, :]
    k = reshape(k, div(m.dim, m.head), m.head, seq_len, :)              # (DIM/HEAD, HEAD, SEQ_LEN, BATCH_SIZE)
    q = reshape(q, div(m.dim, m.head), m.head, seq_len, :)              # (DIM/HEAD, HEAD, SEQ_LEN, BATCH_SIZE)
    v = reshape(v, div(m.dim, m.head), m.head, seq_len, :)              # (DIM/HEAD, HEAD, SEQ_LEN, BATCH_SIZE)

    k_ = permutedims(k, (1, 3, 2, 4))                                   # (DIM/HEAD, SEQ_LEN, HEAD, BATCH_SIZE)
    q = permutedims(q, (3, 1, 2, 4))                                    # (SEQ_LEN, DIM/HEAD, HEAD, BATCH_SIZE)
    v = permutedims(v, (3, 1, 2, 4))                                    # (SEQ_LEN, DIM/HEAD, HEAD, BATCH_SIZE)

    # println("[$(typeof(q))] $(size(q)) * [$(typeof(k_))] $(size(k_))")
    att = Transformers.batchedmul(q, k_)                                # (SEQ_LEN, SEQ_LEN, HEAD, BATCH_SIZE) = (SEQ_LEN, DIM/HEAD, HEAD, BATCH_SIZE) * (DIM/HEAD, SEQ_LEN, HEAD, BATCH_SIZE)
    att = softmax(att ./ sqrt(Float32(m.dim / m.head)), dims=2)         # (SEQ_LEN, SEQ_LEN, HEAD, BATCH_SIZE)

    # TODO: add denoising mask here - potential future work

    # println("[$(typeof(att))] $(size(att)) * [$(typeof(v))] $(size(v))")
    att = Transformers.batchedmul(att, v)                               # (SEQ_LEN, DIM/HEAD, HEAD, BATCH_SIZE) = (SEQ_LEN, SEQ_LEN, HEAD, BATCH_SIZE) * (SEQ_LEN, DIM/HEAD, HEAD, BATCH_SIZE)
    att = permutedims(att, (2, 3, 1, 4))                                # (DIM/HEAD, HEAD, SEQ_LEN, BATCH_SIZE)
    att = reshape(att, m.dim, seq_len, :)                               # (DIM, SEQ_LEN, BATCH_SIZE)

    y = m.output(att)                                                   # (DIM, SEQ_LEN, BATCH_SIZE)
    y = Flux.dropout(y, m.dropout)                                      # (DIM, SEQ_LEN, BATCH_SIZE)

    return y
end

# build model
build_model = (n_head::Int, n_layer::Int; dropout=0.1f0) -> begin

    # construct n_layer of blocks
    blocks = [
        Chain(
            IdentitySkip(
                Chain(
                    CustomAttention(n_head, DIM, dropout=dropout),
                    LayerNorm(DIM),
                )
            ),
            IdentitySkip(
                Chain(
                    Dense(DIM => DIM * 4, gelu),
                    Dense(DIM * 4 => DIM, gelu),
                    Dropout(dropout),
                    LayerNorm(DIM),
                )
            )
        ) for i in 1:n_layer
    ]

    # construct model
    model = Chain(
        Embed(DIM, MOVIE_SIZE + 3),                                     # (SEQ_LEN, BATCH_SIZE) => (DIM, SEQ_LEN, BATCH_SIZE)
        IdentitySkip(
            PositionEmbedding(DIM, trainable=true),                     # (DIM, SEQ_LEN, BATCH_SIZE) => (DIM, SEQ_LEN, BATCH_SIZE)
        ),
        LayerNorm(DIM),
        Chain(blocks...),                                               # n_layer Blocks of CustomAttention
        Dense(DIM => MOVIE_SIZE, gelu),                                 # (VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)
        softmax,
    )

    return model

end


# loss function.  x must have already been padded with preceeding NULL_VALUE
lossF = (model, x, masks) -> begin

    masked_x = x .* (1 .- masks) .+ (masks .* MASK_VALUE)

    y = model(masked_x)                                                 # (VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)

    y_truth = reshape(x .* masks, (1, size(x)...)) .== 1:MOVIE_SIZE     # (VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)
    y_truth = Float32.(y_truth)

    loss_embed = y_truth .* log.(y)
    loss_embed_sum = reshape(-sum(loss_embed, dims=(1, 2)), :)          # (BATCH_SIZE,)

    masks_sum = reshape(1 ./ sum(masks, dims=1), :)                     # (SEQ_LEN, BATCH_SIZE) => (BATCH_SIZE)

    loss_batch = mean(loss_embed_sum .* masks_sum)

    return loss_batch

end

# get fixed sequence lenth data
function get_fix_len_sequence_data(sequences; seq_len=args["seq_len"])
    result = []
    for seq in sequences
        # if less than seq_len, pad and return
        if length(seq) < seq_len
            continue                        # skip any users with less than seq_len ratings
        end
        # if longer than seq_len, sample len(seq)/seq_len times
        for i in 1:seq_len:length(seq)-seq_len+1
            push!(result, seq[i:i+seq_len-1])
        end
    end
    return result
end


train = () -> begin

    # train model
    model = build_model(args["model_nhead"], args["model_nlayer"], dropout=args["model_dropout"])
    if args["model_cuda"] >= 0
        model = model |> gpu
    end
    @info "Model" model

    opt = AdamW(args["train_lr"], (0.9, 0.999), args["train_weight_decay"])
    if args["model_cuda"] >= 0
        opt = opt |> gpu
    end
    @info "Optimizer" opt

    function pad(x, target_len::Int)
        if length(x) < target_len
            return vcat(fill(NULL_VALUE, target_len - length(x)), x)
        else
            return x
        end
    end

    BATCH_SIZE = args["batch_size"]
    NUM_EPOCHS = args["train_epochs"]

    params = Flux.params(model)

    # train epochs
    for epoch in 1:NUM_EPOCHS

        # refresh train data each epoch
        train_data = get_fix_len_sequence_data(TRAIN_SEQUENCES)

        # shuffle train data by batch
        data_loader = Flux.Data.DataLoader(train_data, batchsize=BATCH_SIZE, shuffle=true)

        progress_tracker = Progress(length(data_loader), dt=0.2, desc="Training epoch $(epoch): ")

        loss_list = []

        for batch in data_loader

            # convert vector of sequences to matrix
            batch = reduce(hcat, batch)                                                 # (SEQ_LEN, BATCH_SIZE)

            masks = rand(Float32, size(batch)) .< MASK_RATIO                            # (SEQ_LEN, BATCH_SIZE)
            masks[end, :] .= 1                                                          # mask last item in sequence

            # y_truth = onehotbatch(batch .* masks, 0:MOVIE_SIZE)                       # (VOCAB_SIZE+1, SEQ_LEN, BATCH_SIZE)
            # y_truth = y_truth[2:end, :, :]                                            # (VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)

            # move data to GPU if available
            if args["model_cuda"] >= 0
                batch = batch |> gpu
                masks = masks |> gpu
                # y_truth = y_truth |> gpu
            end

            # @info "Masks" masks
            # @info "Batch" batch

            loss, back = pullback(params) do
                lossF(model, batch, masks)
            end

            grads = back(1.0f0)

            Flux.update!(opt, params, grads)

            push!(loss_list, loss)
            loss_avg = round(mean(loss_list), digits=3)

            next!(progress_tracker, showvalues=[
                :loss => loss_avg,
                :curr => round(loss, digits=3),
            ])

        end

        # GC and reclaim GPU memory after each epoch
        GC.gc(false)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end

    end

    return model

end

##################################################
# save
function save_model(model)
    model_filename = "trained/bert_$(args["model_dim"])x$(args["seq_len"])_h$(args["model_nhead"])_l$(args["model_nlayer"]).model"
    model_ = model |> cpu
    @info "Saving model to [$(model_filename)]"
    open(model_filename, "w") do io
        serialize(io, model_)
    end
    return nothing
end

##################################################
# load
function load_model()
    model_filename = "trained/bert_$(args["model_dim"])x$(args["seq_len"])_h$(args["model_nhead"])_l$(args["model_nlayer"]).model"
    @info "Loading model from [$(model_filename)]"
    open(model_filename, "r") do io
        model_ = deserialize(io)
        # Flux.loadmodel!(model, model_)
        if args["model_cuda"] >= 0
            model_ = model_ |> gpu
        end
        return model_
    end
end


if abspath(PROGRAM_FILE) == @__FILE__

    model = train()

    save_model(model)

end
