# VisionTransformers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaBillson.github.io/VisionTransformers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaBillson.github.io/VisionTransformers.jl/dev/)
[![Build Status](https://github.com/JoshuaBillson/VisionTransformers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaBillson/VisionTransformers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoshuaBillson/VisionTransformers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaBillson/VisionTransformers.jl)

VisionTransformers is a pure Julia package implementing various vision transformer models in Flux.

# Available Models

| Model    | Source                                                                                                                                        | Implemented        |
| :------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------: |
| ViT      | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)                       | :white_check_mark: |
| PVT      | [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://doi.org/10.48550/arXiv.2102.12122)       | :white_check_mark: |
| CvT      | [CvT: Introducing Convolutions to Vision Transformers](https://doi.org/10.48550/arXiv.2103.15808)                                             | :white_check_mark: |
| SWIN     | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://doi.org/10.48550/arXiv.2103.14030)                          | :white_check_mark: |
| Twins    | [Twins: Revisiting the Design of Spatial Attention in Vision Transformers](https://doi.org/10.48550/arXiv.2104.13840)                         | :x:                |
| CaiT     | [Going deeper with Image Transformers](https://doi.org/10.48550/arXiv.2103.17239)                                                             | :x:                |
