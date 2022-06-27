#pragma once

#include "ScreenPass.h"

struct FMotionVectorAndUsedCountOutput {
    FRDGTextureRef DilatedVelocityOutput = nullptr;
    FRDGTextureRef ClosestDepthOutput = nullptr;
    FRDGTextureRef PrevClosestDepthOutput = nullptr;
    FRDGTextureRef PrevUseCountOutput = nullptr;
};

FRDGTextureRef AddMotionVectorAndUsedCountPass(
    FRDGBuilder& GraphBuilder,
    const FViewInfo& View,
    FRDGTextureRef SceneDepthTexture,
    FRDGTextureRef SceneVelocityTexture
);

//FRDGTextureRef AddParallaxRejectionMaskPass(
//    FRDGBuilder& GraphBuilder,
//    const FViewInfo& View,
//    FRDGTextureRef DilatedVelocityTexture,
//    FRDGTextureRef ClosestDepthTexture,
//    FRDGTextureRef PrevClosestDepthTexture,
//    FRDGTextureRef PrevUseCountTexture
//);
