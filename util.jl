using Flux

##################################################
"""
Identity Skip function.  E.g.

IdentitySkip(
    Chain(
        LayerNorm(DIM),
        Dense(DIM => DIM * 2),
        Dense(DIM * 2 => DIM),
    )
),
"""
struct IdentitySkip
    inner
end

Flux.@functor IdentitySkip

(m::IdentitySkip)(x) = begin
    # println("IdentitySkip $(size(x))")
    x .+ m.inner(x)
end

##################################################
"""
Custom Reshape layer.  E.g.

    Reshape(3, 3, 1, :)
"""
struct Reshape
    shape
end

Flux.@functor Reshape

Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)


##################################################
"""
Custom split layer.  E.g.

Split(
    Dense(5 => 5, tanh),    # policy 1 output
    Dense(5 => 5, tanh),    # policy 2 output
    Dense(5 => 1),          # value output
)
"""
struct Split{T}
    paths::T
end

Flux.@functor Split

Split(paths...) = Split(paths)
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

##################################################
"""
Custom join layer.  e.g.

Join(
    vcat,
    Chain(
        Dense(1 => 5, relu),
        Dense(5 => 1)
    ,
    Dense(1 => 2),
    Dense(1 => 1)
),
"""
# custom join layer
struct Join{T,F}
    combine::F
    paths::T
end

Flux.@functor Join

Join(combine, paths...) = Join(combine, paths)

(m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
(m::Join)(xs...) = m(xs)

# Join(combine, paths) = Parallel(combine, paths)
