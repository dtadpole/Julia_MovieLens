include("./bert4rec.jl")

using Recommendation

eval_model = (model, input_sequences; label="Evaluation") -> begin

    PRED_K = 10

    eval_data = get_fix_len_sequence_data(input_sequences)

    # shuffle train data by batch
    data_loader = Flux.Data.DataLoader(eval_data, batchsize=args["batch_size"], shuffle=true)

    progress_tracker = Progress(length(data_loader), dt=0.2, desc="$(label): ")

    recall_1_list = []
    recall_5_list = []
    recall_10_list = []
    ndcg_5_list = []
    ndcg_10_list = []

    for batch in data_loader

        # convert vector of sequences to matrix
        batch = reduce(hcat, batch)                                 # (SEQ_LEN, BATCH_SIZE)
        batch_size = size(batch, 2)
        if args["model_cuda"] >= 0
            batch = batch |> gpu
        end

        masks = zeros(Int, size(batch))                             # (SEQ_LEN, BATCH_SIZE)
        masks[size(masks, 1), :] .= 1                               # mask the last item in sequence
        if args["model_cuda"] >= 0
            masks = masks |> gpu
        end

        masked_batch = batch .* (1 .- masks) .+ (masks .* MASK_VALUE)

        y = model(masked_batch)                                     # (VOCAB_SIZE, SEQ_LEN, BATCH_SIZE)

        batch_recall_1 = 0.0
        batch_recall_5 = 0.0
        batch_recall_10 = 0.0
        batch_ndcg_1 = 0.0
        batch_ndcg_5 = 0.0
        batch_ndcg_10 = 0.0

        for i in 1:batch_size

            truth = [batch[end, i]]

            pred = reshape(y[:, end, i], :)                         # (VOCAB_SIZE, batch_size)
            pred_top_k = Vector(partialsortperm(pred, 1:PRED_K, rev=true))

            # println("Actual: $(truth)")
            # println("Predicted: $(pred_top_k)")

            batch_recall_1 += measure(Recall(), truth, pred_top_k, 1)
            batch_recall_5 += measure(Recall(), truth, pred_top_k, 5)
            batch_recall_10 += measure(Recall(), truth, pred_top_k, 10)
            batch_ndcg_5 += measure(NDCG(), truth, pred_top_k, 5)
            batch_ndcg_10 += measure(NDCG(), truth, pred_top_k, 10)

        end

        batch_recall_1 /= batch_size
        batch_recall_5 /= batch_size
        batch_recall_10 /= batch_size
        batch_ndcg_5 /= batch_size
        batch_ndcg_10 /= batch_size

        push!(recall_1_list, batch_recall_1)
        push!(recall_5_list, batch_recall_5)
        push!(recall_10_list, batch_recall_10)
        push!(ndcg_5_list, batch_ndcg_5)
        push!(ndcg_10_list, batch_ndcg_10)

        next!(progress_tracker, showvalues=[
            :recall_1 => round(mean(recall_1_list), digits=3),
            :recall_5 => round(mean(recall_5_list), digits=3),
            :recall_10 => round(mean(recall_10_list), digits=3),
            :ndcg_5 => round(mean(ndcg_5_list), digits=3),
            :ndcg_10 => round(mean(ndcg_10_list), digits=3),
            :recall_1_batch => round(batch_recall_1, digits=3),
            :recall_5_batch => round(batch_recall_5, digits=3),
            :recall_10_batch => round(batch_recall_10, digits=3),
            :ndcg_5_batch => round(batch_ndcg_5, digits=3),
            :ndcg_10_batch => round(batch_ndcg_10, digits=3),
        ])

    end

    # GC and reclaim GPU memory after each epoch
    GC.gc(false)
    if args["model_cuda"] >= 0
        CUDA.reclaim()
    end

end


if abspath(PROGRAM_FILE) == @__FILE__

    model = load_model()
    @info "Loaed model" model

    @info "Evaluating model with validation data"
    eval_model(model, VALIDATE_SEQUENCES, label="Validation Data")

    @info "Evaluating model with test data"
    eval_model(model, TEST_SEQUENCES, label="Test Data")

    @info "Evaluating model with training data"
    eval_model(model, TRAIN_SEQUENCES, label="Train Data")

end
