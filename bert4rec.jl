include("./data.jl")
include("./util.jl")

using Distributions
using Transformers
using Transformers.Basic
using ProgressMeter: Progress, next!, finish!
using Zygote: pullback
using OneHotArrays

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

DIM = 32
SEQ_LEN = 20
BATCH_SIZE = 64
NUM_EPOCHS = 10

MASK_RATIO = 0.2

# custom attention layer (for future experiments)
struct CustomAttention
    # attributes
    head::Int
    dim::Int
    input::Dense
    output::Dense
    dropout::Float32
    # constructor
    function CustomAttention(head, dim; dropout=0.1f0)
        if dim % head != 0
            error("Dimension must be divisible by head")
        end
        new(head, dim, Dense(dim => dim * 3), Dense(dim => dim), dropout)
    end
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
        Dense(DIM => MOVIE_SIZE + 3, gelu),                             # (DIM, SEQ_LEN, BATCH_SIZE) => (MOVIE_SIZE + 3, SEQ_LEN, BATCH_SIZE)
        softmax,
    )

    return model

end


# loss function.  x must have already been padded with preceeding NULL_VALUE
lossF = (x, masks) -> begin

    masked_x = x .* (1 .- masks) .+ (masks .* MASK_VALUE)

    y = model(masked_x)

    y_truth = onehotbatch(x .* masks, 1:MOVIE_SIZE+3, MOVIE_SIZE + 3)   # TODO

    loss_embed = -sum(y_truth .* log.(y), dims=1)

    loss_embed_sum = reshape(sum(loss_embed, dims=(1, 2)), :)

    masks_sum = reshape(1 ./ sum(masks, dims=1), :)

    loss_seq = loss_embed_sum .* masks_sum

    loss_batch = mean(loss_seq)

    return loss_batch

end

# train model
model = build_model(4, 2)
@info "Model" model

opt = AdamW(0.001, (0.9, 0.999), 0.0001)
@info "Optimizer" opt

train = () -> begin

    function pad(x, target_len::Int)
        if length(x) < target_len
            return vcat(fill(NULL_VALUE, target_len - length(x)), x)
        else
            return x
        end
    end

    function get_train_data()
        result = []
        for seq in TRAIN_SEQUENCES
            # if less than SEQ_LEN, pad and return
            if length(seq) < SEQ_LEN / 2
                continue                        # skip any users with less than SEQ_LEN / 2 ratings
            elseif length(seq) < SEQ_LEN
                seq = pad(seq, SEQ_LEN)         # pad with NULL_VALUE
                push!(result, seq)
                continue
            end
            # if longer than SEQ_LEN, sample len(seq)/SEQ_LEN times
            indices = sample(1:length(seq)-SEQ_LEN+1, div(length(seq), SEQ_LEN), replace=false)
            for i in indices
                push!(result, seq[i:i+SEQ_LEN-1])
            end
        end

        return result
    end

    params = Flux.params(model)

    # train epochs
    for epoch in 1:NUM_EPOCHS

        # refresh train data each epoch
        train_data = get_train_data()

        # shuffle train data by batch
        data_loader = Flux.Data.DataLoader(train_data, batchsize=BATCH_SIZE, shuffle=true)

        progress_tracker = Progress(length(data_loader), dt=0.2, desc="Training epoch $(epoch): ")

        loss_list = []

        for batch in data_loader

            # convert vector of sequences to matrix
            batch = reduce(hcat, batch)                         # (SEQ_LEN, BATCH_SIZE)

            masks = rand(Float32, size(batch)) .< MASK_RATIO    # (SEQ_LEN, BATCH_SIZE)
            masks[size(masks, 1), :] .= 1                       # always mask last item in sequence

            # @info "Masks" masks
            # @info "Batch" batch

            loss, back = pullback(params) do
                lossF(batch, masks)
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
    end

end


if abspath(PROGRAM_FILE) == @__FILE__

    train()

end
