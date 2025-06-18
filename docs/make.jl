using VisionTransformers
using Documenter

DocMeta.setdocmeta!(VisionTransformers, :DocTestSetup, :(using VisionTransformers); recursive=true)

makedocs(;
    modules=[VisionTransformers],
    authors="Joshua Billson",
    sitename="VisionTransformers.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/VisionTransformers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/VisionTransformers.jl",
    devbranch="main",
)
