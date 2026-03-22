```@meta
CurrentModule = VisionTransformers
```

# VisionTransformers

[VisionTransformers](https://github.com/JoshuaBillson/VisionTransformers.jl) is a pure Julia package implementing various vision transformer models in Flux.


# Available Models

| Model    | Source                                                                                                                                        | Implemented        |
| :------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: |
| ViT      | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)                       | :white_check_mark: |
| PVT      | [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://doi.org/10.48550/arXiv.2102.12122)       | :white_check_mark: |
| CvT      | [CvT: Introducing Convolutions to Vision Transformers](https://doi.org/10.48550/arXiv.2103.15808)                                             | :white_check_mark: |
| SWIN     | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://doi.org/10.48550/arXiv.2103.14030)                          | :white_check_mark: |
| Twins    | [Twins: Revisiting the Design of Spatial Attention in Vision Transformers](https://doi.org/10.48550/arXiv.2104.13840)                         |  :white_check_mark: |
| CaiT     | [Going deeper with Image Transformers](https://doi.org/10.48550/arXiv.2103.17239)                                                             | :x:                |


# Models

```@docs
ViT
CvT
PVT
SWIN
Twins
```

# Layers
```@docs
MultiHeadAttention
ConvAttention
SRAttention
WindowedAttention
AbsolutePositionEmbedding
VariablePositionEmbedding
RelativePositionEmbedding
PEG
MLP
SeparableConv
Tokens
StripTokens
```

# Utilities
```@docs
img2seq
seq2img
```
