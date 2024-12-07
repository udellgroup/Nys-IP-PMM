data2methods = Dict(
    # "CIFAR10_1000" => Dict(
    #     "method_Ps" => [method_Nystrom(200, false)]
    # ),
    # "sector" => Dict( 
    #     "method_Ps"   => [method_Nystrom(20, false)]#[method_PartialCholesky(20), method_Nystrom(20, false), method_NoPreconditioner()]
    # ),
    # "dexter" => Dict(
    #     "method_Ps" => [method_Nystrom(10, false)]#[method_PartialCholesky(10), method_Nystrom(10, false), method_NoPreconditioner()]
    # ),
    # "arcene" => Dict(
    #     "method_Ps" => [method_Nystrom(20, false)]#[method_PartialCholesky(20), method_Nystrom(20, false), method_NoPreconditioner()]
    # ),
    # "RNASeq" => Dict(
    #     "method_Ps"   => [method_Nystrom(200, false)]#[method_PartialCholesky(200), method_Nystrom(200, false), method_NoPreconditioner()]
    # ),
    "STL10" => Dict(
        "method_Ps"   => [method_Nystrom(800, false)]#[method_PartialCholesky(800), method_Nystrom(800, false), method_NoPreconditioner()]
    ),
    # "CIFAR10" => Dict(
    #     "method_Ps"   => [method_Nystrom(200, false)]#[method_PartialCholesky(200), method_Nystrom(200, false), method_NoPreconditioner()]
    # ),
    # "SensIT" => Dict(
    #     "method_Ps"   => [method_Nystrom(50, false)]#[method_PartialCholesky(50), method_Nystrom(50, false), method_NoPreconditioner()]
    # )
)