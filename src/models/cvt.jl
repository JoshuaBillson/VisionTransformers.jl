"""
    CvT(config::Symbol; kw...)
    CvT(dim::Int, depths, nheads;
        inchannels=3, mlp_ratio=4,
        dropout=0.1, drop_path=0.0,
        nclasses=1000)

Construct a Convolutional Vision Transformer (CvT) model for image
classification. CvT extends the Vision Transformer (ViT) by replacing
linear projections with convolutional projections and by introducing
convolutional token embeddings. This design enhances locality and
translation invariance while retaining the benefits of Transformer-based
global modeling.

# Arguments
- `config`: One of `:B13`, `:B21`, or `:W24`.
- `dim`: Embedding dimension of patch tokens in the first stage.
- `depths`: Number of Transformer blocks in each stage.
- `nheads`: Number of attention heads in each stage.
- `inchannels`: Number of input image channels. Default is `3`.
- `mlp_ratio:`: Expansion ratio for the hidden dimension of the MLP relative to `dim`. Default is `4`.
- `dropout`: Dropout probability applied to embeddings, MLP, and attention outputs.
- `drop_path`: Probability for stochastic depth (drop-path) regularization.
- `nclasses`: Number of output classes for classification. Default is `1000`.
"""
function CvT(config::Symbol; kw...)
    @match config begin
        :B13 => CvT(64, [1,2,10], [1,3,6]; kw...)
        :B21 => CvT(64, [1,4,16], [1,3,6], kw...)
        :W24 => CvT(64, [2,2,20], [3,12,16], kw...)
    end
end

function CvT(dim::Int, depths, nheads; inchannels=3, mlp_ratio=4, dropout=0.1, drop_path=0.0, nclasses=1000)
    @argcheck length(depths) == length(nheads) == 3

    # Get Per-Block Path Drop Rate
    drop_path_rates = _per_layer_drop_path(drop_path, depths)

    Flux.Chain(
        # Encoder
        Flux.Chain(

            # Stage 1
            CvTStage(
                inchannels, 
                dim*nheads[1], 
                depths[1]; 
                nheads=nheads[1], 
                patchsize=7, 
                patchstride=4, 
                qkv_bias=true, 
                drop_path=drop_path_rates[1],
                mlp_ratio, 
                dropout, 
            ),

            # Stage 2
            CvTStage(
                dim*nheads[1], 
                dim*nheads[2], 
                depths[2]; 
                nheads=nheads[2], 
                qkv_bias=true, 
                drop_path=drop_path_rates[2],
                mlp_ratio, 
                dropout, 
            ),

            # Stage 3
            CvTStage(
                dim*nheads[2], 
                dim*nheads[3], 
                depths[3]; 
                nheads=nheads[3], 
                qkv_bias=true, 
                drop_path=drop_path_rates[3],
                mlp_ratio, 
                dropout, 
            ),
        ), 

        # Classification Head
        Flux.Chain(
            Base.Fix2(permutedims, (3,1,2,4)),
            Flux.LayerNorm(dim*nheads[3]), 
            x -> dropdims(mean(x, dims=(2,3)), dims=(2,3)),
            Flux.Dense(dim*nheads[3], nclasses),
        )
    )
end

function CvTStage(inplanes, outplanes, depth; nheads=4, patchsize=3, patchstride=2, kernel_size=3, mlp_ratio=4, qkv_bias=false, q_stride=1, kv_stride=2, dropout=0., drop_path=0.)
    drop_path = drop_path isa Number ? repeat([drop_path], depth) : drop_path
    Flux.Chain(
        Flux.Conv((patchsize,patchsize), inplanes=>outplanes, stride=(patchstride,patchstride), pad=Flux.SamePad()),
        Base.Fix2(permutedims, (3,1,2,4)),
        Flux.LayerNorm(outplanes),
        Flux.Chain([CvTBlock(outplanes; nheads, kernel_size, mlp_ratio, qkv_bias, q_stride, kv_stride, dropout, drop_path=drop_path[i]) for i in 1:depth]...),
        Base.Fix2(permutedims, (2,3,1,4)),
    )
end

function CvTBlock(dim; nheads=4, kernel_size=3, mlp_ratio=4, qkv_bias=false, q_stride=1, kv_stride=1, dropout=0., drop_path=0.)
    Flux.Chain(
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim),
                ConvAttention(
                    dim; 
                    kernel=(kernel_size,kernel_size),
                    q_stride=(q_stride,q_stride),
                    kv_stride=(kv_stride,kv_stride),
                    attn_dropout_prob=dropout, 
                    proj_dropout_prob=dropout, 
                    nheads, 
                    qkv_bias
                ),
                Flux.Dropout(drop_path, dims=4)
            ), 
            +
        ), 
        Flux.SkipConnection(
            Flux.Chain(
                Flux.LayerNorm(dim), 
                MLP(dim, dim*mlp_ratio, dim; dropout),
                Flux.Dropout(drop_path, dims=4)
            ), 
            +
        )
    )
end