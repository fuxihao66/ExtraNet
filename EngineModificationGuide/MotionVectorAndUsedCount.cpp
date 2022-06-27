// Calculate MotionVector and get prev used count for mask hole

#include "PostProcess/MotionVectorAndUsedCount.h"
#include "PostProcess/PostProcessing.h"

#include "PostProcessing.h"
#include "SceneTextureParameters.h"
#include "PixelShaderUtils.h"

//class FClearPrevTexturesCS : public FGlobalShader {
//public:
//    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
//        return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
//    }
//
//    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment) {
//        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
//    }
//
//    DECLARE_GLOBAL_SHADER(FClearPrevTexturesCS);
//    SHADER_USE_PARAMETER_STRUCT(FClearPrevTexturesCS, FGlobalShader);
//
//    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
//
//
//        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
//        SHADER_PARAMETER_STRUCT(FScreenPassTextureViewportParameters, InputInfo)
//
//        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D, PrevUseCountOutput)
//        SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D, PrevClosestDepthOutput)
//        END_SHADER_PARAMETER_STRUCT()
//};

class FMotionVectorAndUsedCountCS : public FGlobalShader {
public:
    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
        return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
    }

    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment) {
        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
    }

    DECLARE_GLOBAL_SHADER(FMotionVectorAndUsedCountCS);
    SHADER_USE_PARAMETER_STRUCT(FMotionVectorAndUsedCountCS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        // Input images
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneDepthTexture)
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneVelocityTexture)

        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
        SHADER_PARAMETER_STRUCT(FScreenPassTextureViewportParameters, InputInfo)

        // Output images
        SHADER_PARAMETER_RDG_TEXTURE_UAV(Texture2D, DilatedVelocityOutput)
        //SHADER_PARAMETER_RDG_TEXTURE_UAV(Texture2D, ClosestDepthOutput)
        //SHADER_PARAMETER_RDG_TEXTURE_UAV(Texture2D, PrevUseCountOutput)
        //SHADER_PARAMETER_RDG_TEXTURE_UAV(Texture2D, PrevClosestDepthOutput)

        END_SHADER_PARAMETER_STRUCT()

};

//class FParallaxRejectionMaskCS : public FGlobalShader {
//public:
//    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
//        return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
//    }
//
//    static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment) {
//        FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
//    }
//
//    DECLARE_GLOBAL_SHADER(FParallaxRejectionMaskCS);
//    SHADER_USE_PARAMETER_STRUCT(FParallaxRejectionMaskCS, FGlobalShader);
//
//    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
//        // Input images
//        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, DilatedVelocityTexture)
//        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, ClosestDepthTexture)
//        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, PrevUseCountTexture)
//        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, PrevClosestDepthTexture)
//
//        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
//        SHADER_PARAMETER_STRUCT(FScreenPassTextureViewportParameters, InputInfo)
//
//        // Output images
//        SHADER_PARAMETER_RDG_TEXTURE_UAV(Texture2D, ParallaxRejectionMaskOutput)
//
//        END_SHADER_PARAMETER_STRUCT()
//
//};

//class FDebugCS : public FGlobalShader {
//public:
//	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) {
//		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
//	}
//
//	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment) {
//		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
//	}
//
//	DECLARE_GLOBAL_SHADER(FDebugCS);
//	SHADER_USE_PARAMETER_STRUCT(FDebugCS, FGlobalShader);
//
//	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
//		// Input images
//		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, PrevUseCountTexture)
//
//		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
//		SHADER_PARAMETER_STRUCT(FScreenPassTextureViewportParameters, InputInfo)
//
//		// Output images
//		SHADER_PARAMETER_RDG_TEXTURE_UAV(Texture2D, DebugOutput)
//
//		END_SHADER_PARAMETER_STRUCT()
//
//};

//IMPLEMENT_GLOBAL_SHADER(FClearPrevTexturesCS, "/Engine/Private/ClearPrevTextures.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FMotionVectorAndUsedCountCS, "/Engine/Private/MotionVectorAndCountUsed.usf", "MainCS", SF_Compute);
//IMPLEMENT_GLOBAL_SHADER(FParallaxRejectionMaskCS, "/Engine/Private/ParallaxRejectionMask.usf", "MainCS", SF_Compute);
// IMPLEMENT_GLOBAL_SHADER(FDebugCS, "/Engine/Private/DebugCountShader.usf", "MainCS", SF_Compute);



FRDGTextureRef AddMotionVectorAndUsedCountPass(
    FRDGBuilder& GraphBuilder,
    const FViewInfo& View,
    FRDGTextureRef SceneDepthTexture,
    FRDGTextureRef SceneVelocityTexture
) {
    FRDGTextureRef DilatedVelocityTexture;
    //FRDGTextureRef ClosestDepthTexture;
    //FRDGTextureRef PrevUseCountTexture;
    //FRDGTextureRef PrevClosestDepthTexture;

    //FRDGTextureDesc Desc = FRDGTextureDesc::Create2DDesc(
    //    SceneVelocityTexture->Desc.Extent,
    //    PF_R32_UINT,
    //    FClearValueBinding::None,
    //    /* InFlags = */ TexCreate_None,
    //    /* InTargetableFlags = */ TexCreate_ShaderResource | TexCreate_UAV,
    //    /* bInForceSeparateTargetAndShaderResource = */ false);

    //PrevUseCountTexture = GraphBuilder.CreateTexture(Desc, TEXT("PrevUseCountTexture"));
    //PrevClosestDepthTexture = GraphBuilder.CreateTexture(Desc, TEXT("PrevClosestDepthTexture"));

    FRDGTextureDesc FloatDesc = FRDGTextureDesc::Create2DDesc(
        SceneVelocityTexture->Desc.Extent,
        PF_FloatRGBA,
        FClearValueBinding::Black,
        /* InFlags = */ TexCreate_None,
        /* InTargetableFlags = */ TexCreate_ShaderResource | TexCreate_UAV,
        /* bInForceSeparateTargetAndShaderResource = */ false);

    DilatedVelocityTexture = GraphBuilder.CreateTexture(FloatDesc, TEXT("DilatedVelocityTexture"), ERDGResourceFlags::MultiFrame);
    //ClosestDepthTexture = GraphBuilder.CreateTexture(FloatDesc, TEXT("ClosestDepthTexture"));

  //  FClearPrevTexturesCS::FParameters* ClearPassParameters = GraphBuilder.AllocParameters<FClearPrevTexturesCS::FParameters>();

  //  ClearPassParameters->View = View.ViewUniformBuffer;
  //  ClearPassParameters->InputInfo = GetScreenPassTextureViewportParameters(FScreenPassTextureViewport(SceneVelocityTexture->Desc.Extent, View.ViewRect));


  //  ClearPassParameters->PrevUseCountOutput = GraphBuilder.CreateUAV(PrevUseCountTexture);
  //  ClearPassParameters->PrevClosestDepthOutput = GraphBuilder.CreateUAV(PrevClosestDepthTexture);

  //  TShaderMapRef<FClearPrevTexturesCS> PrevClearComputeShader(View.ShaderMap);
  //  FComputeShaderUtils::AddPass(
  //      GraphBuilder,
  //      RDG_EVENT_NAME("ClearPrevTextures"),
		//PrevClearComputeShader,
  //      ClearPassParameters,
  //      FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), 8));

    FMotionVectorAndUsedCountCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMotionVectorAndUsedCountCS::FParameters>();

    PassParameters->SceneDepthTexture = SceneDepthTexture;
    PassParameters->SceneVelocityTexture = SceneVelocityTexture;

    PassParameters->View = View.ViewUniformBuffer;
    PassParameters->InputInfo = GetScreenPassTextureViewportParameters(FScreenPassTextureViewport(SceneVelocityTexture->Desc.Extent, View.ViewRect));

    PassParameters->DilatedVelocityOutput = GraphBuilder.CreateUAV(DilatedVelocityTexture);
    //PassParameters->ClosestDepthOutput = GraphBuilder.CreateUAV(ClosestDepthTexture);
    //PassParameters->PrevUseCountOutput = GraphBuilder.CreateUAV(PrevUseCountTexture);
    //PassParameters->PrevClosestDepthOutput = GraphBuilder.CreateUAV(PrevClosestDepthTexture);

    TShaderMapRef<FMotionVectorAndUsedCountCS> ComputeShader(View.ShaderMap);

    FComputeShaderUtils::AddPass(
        GraphBuilder,
        RDG_EVENT_NAME("DilateVelocity"),
        ComputeShader,
        PassParameters,
        FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), 8));

    //FMotionVectorAndUsedCountOutput Output;
    //Output.DilatedVelocityOutput = DilatedVelocityTexture;
    //Output.ClosestDepthOutput = ClosestDepthTexture;
    //Output.PrevClosestDepthOutput = PrevUseCountTexture;
    //Output.PrevUseCountOutput = PrevClosestDepthTexture;

    return DilatedVelocityTexture;
}

//FRDGTextureRef AddParallaxRejectionMaskPass(
//    FRDGBuilder& GraphBuilder,
//    const FViewInfo& View,
//    FRDGTextureRef DilatedVelocityTexture,
//    FRDGTextureRef ClosestDepthTexture,
//    FRDGTextureRef PrevUseCountTexture,
//    FRDGTextureRef PrevClosestDepthTexture
//) {
//    FRDGTextureRef ParallaxRejectionMaskTexture;
//
//    FRDGTextureDesc FloatDesc = FRDGTextureDesc::Create2DDesc(
//        DilatedVelocityTexture->Desc.Extent,
//        PF_FloatRGBA,
//        FClearValueBinding::Black,
//        /* InFlags = */ TexCreate_None,
//        /* InTargetableFlags = */ TexCreate_ShaderResource | TexCreate_UAV,
//        /* bInForceSeparateTargetAndShaderResource = */ false);
//
//    ParallaxRejectionMaskTexture = GraphBuilder.CreateTexture(FloatDesc, TEXT("ParallaxRejectionMaskTexture"));
//
//    FParallaxRejectionMaskCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FParallaxRejectionMaskCS::FParameters>();
//
//    PassParameters->DilatedVelocityTexture = DilatedVelocityTexture;
//    PassParameters->ClosestDepthTexture = ClosestDepthTexture;
//    PassParameters->PrevUseCountTexture = PrevUseCountTexture;
//    PassParameters->PrevClosestDepthTexture = PrevClosestDepthTexture;
//
//    PassParameters->View = View.ViewUniformBuffer;
//    PassParameters->InputInfo = GetScreenPassTextureViewportParameters(FScreenPassTextureViewport(DilatedVelocityTexture->Desc.Extent, View.ViewRect));
//
//    PassParameters->ParallaxRejectionMaskOutput = GraphBuilder.CreateUAV(ParallaxRejectionMaskTexture);
//
//    TShaderMapRef<FParallaxRejectionMaskCS> ComputeShader(View.ShaderMap);
//
//    FComputeShaderUtils::AddPass(
//        GraphBuilder,
//        RDG_EVENT_NAME("ParallaxRejectionMask"),
//        ComputeShader,
//        PassParameters,
//        FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), 8));
//
//    return ParallaxRejectionMaskTexture;
//}