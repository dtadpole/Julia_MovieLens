using ArgParse
using CUDA

##################################################
# Parse command line arguments
function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin

        "--model_cuda"
        help = "model cuda number"
        arg_type = Int
        default = -1

        "--model_dim"
        help = "model embedding dimension"
        arg_type = Int
        default = 128

        "--model_nhead"
        help = "model number of heads"
        arg_type = Int
        default = 4

        "--model_nlayer"
        help = "model number of layers"
        arg_type = Int
        default = 2

        "--model_dropout"
        help = "model dropout"
        arg_type = Float32
        default = 0.2f0

        "--batch_size"
        help = "batch size"
        arg_type = Int
        default = 64

        "--train_epochs"
        help = "train epochs"
        arg_type = Int
        default = 5

        "--train_lr"
        help = "learning rate"
        arg_type = Float64
        default = 0.002

        "--train_weight_decay"
        help = "weight decay"
        arg_type = Float32
        default = 0.00005f0

        "--seq_len"
        help = "sequence length"
        arg_type = Int
        default = 20

        "--mask_ratio"
        help = "mask ratio"
        arg_type = Float32
        default = 0.2f0

    end

    return parse_args(s)
end

args = parse_commandline()

if args["model_cuda"] >= 0
    CUDA.allowscalar(false)
    CUDA.device!(args["model_cuda"])
end
