using VisionTransformers
using Test

function test_model(model; input_size=(224,224,3,1), output_size=(1000,1))
    x = rand(Float32, input_size)
    y = model(x)
    @test size(y) == output_size
end

@testset "VisionTransformers.jl" begin
    # Test ViT
    test_model(ViT(:tiny))

    # Test CvT
    test_model(CvT(:B13))

    # Test PVT
    test_model(PVT(:tiny))

    # Test SWIN
    test_model(SWIN(:tiny))
end
