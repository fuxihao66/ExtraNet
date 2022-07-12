# How to use higher version of Unreal Engine to export data

If you want to use scenes that only support higher version of Unreal Engine (like 4.27), you can try to follow the instructions below to modify the engine by yourselves.

0. Clone the official version of UE from github.
1. Move ".ush" and ".usf" files to the Engine/Shaders/private folder
2. Move "MotionVectorAndUsedCount.cpp" and "MotionVectorAndUsedCount.h" files to Engine/Source/Runtime/Renderer/Private/PostProcess folder
3. Run the **Setup** and **GenerateProject** scripts
4. Move ".uasset" files in ExtraNet/EngineModificationGuide/BufferVisualization to Engine/Content/BufferVisualization folder
5. Add the following lines to BaseEngine.ini
```
WorldPosition=(Material="/Engine/BufferVisualization/WorldPosition.WorldPosition", Name=LOCTEXT("WorldPosition", "World Position"))
MyStencil=(Material="/Engine/BufferVisualization/MyStencil.MyStencil", Name=LOCTEXT("MyStencil", "Custom Stencil Value"))
NoV=(Material="/Engine/BufferVisualization/NoV.NoV", Name=LOCTEXT("NoV", "NoV"))
MotionVector=(Material="/Engine/BufferVisualization/MotionVector.MotionVector", Name=LOCTEXT("MotionVector", "MotionVector"))
```
6. Modify the AddVisualizeGBufferOverviewPass function in Engine/Source/Runtime/Renderer/Private/PostProcess/PostProcessVisualizeBuffer.cpp:
```
#include "PostProcess/MotionVectorAndUsedCount.h"

...

FScreenPassTexture AddVisualizeGBufferOverviewPass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FVisualizeGBufferOverviewInputs& Inputs)
{
	const FFinalPostProcessSettings& PostProcessSettings = View.FinalPostProcessSettings;

	check(Inputs.SceneColor.IsValid());
	check(Inputs.bDumpToFile || Inputs.bOverview || PostProcessSettings.BufferVisualizationPipes.Num() > 0);


	FScreenPassTexture Output;
	const EPixelFormat OutputFormat = Inputs.bOutputInHDR ? PF_FloatRGBA : PF_Unknown;

	TArray<FVisualizeBufferTile> Tiles;

	RDG_EVENT_SCOPE(GraphBuilder, "VisualizeGBufferOverview");

	const FString& BaseFilename = PostProcessSettings.BufferVisualizationDumpBaseFilename;

	for (UMaterialInterface* MaterialInterface : PostProcessSettings.BufferVisualizationOverviewMaterials)
	{
		if (!MaterialInterface)
		{
			// Add an empty tile to keep the location of each static on the grid.
			Tiles.Emplace();
			continue;
		}

		const FString MaterialName = MaterialInterface->GetName();

		RDG_EVENT_SCOPE(GraphBuilder, "%s", *MaterialName);

		FPostProcessMaterialInputs PostProcessMaterialInputs;
		PostProcessMaterialInputs.SetInput(EPostProcessMaterialInput::SceneColor, Inputs.SceneColor);
		PostProcessMaterialInputs.SetInput(EPostProcessMaterialInput::SeparateTranslucency, Inputs.SeparateTranslucency);
		PostProcessMaterialInputs.SetInput(EPostProcessMaterialInput::PreTonemapHDRColor, Inputs.SceneColorBeforeTonemap);
		PostProcessMaterialInputs.SetInput(EPostProcessMaterialInput::PostTonemapHDRColor, Inputs.SceneColorAfterTonemap);
		PostProcessMaterialInputs.SetInput(EPostProcessMaterialInput::Velocity, Inputs.Velocity);
		PostProcessMaterialInputs.OutputFormat = OutputFormat;

		Output = AddPostProcessMaterialPass(GraphBuilder, View, PostProcessMaterialInputs, MaterialInterface);

		const TSharedPtr<FImagePixelPipe, ESPMode::ThreadSafe>* OutputPipe = PostProcessSettings.BufferVisualizationPipes.Find(MaterialInterface->GetFName());

		if (OutputPipe && OutputPipe->IsValid())
		{
			AddDumpToPipePass(GraphBuilder, Output, OutputPipe->Get());
		}

		if (Inputs.bDumpToFile)
		{
			// First off, allow the user to specify the pass as a format arg (using {material})
			TMap<FString, FStringFormatArg> FormatMappings;
			FormatMappings.Add(TEXT("material"), MaterialName);

			FString MaterialFilename = FString::Format(*BaseFilename, FormatMappings);

			// If the format made no change to the string, we add the name of the material to ensure uniqueness
			if (MaterialFilename == BaseFilename)
			{
				MaterialFilename = BaseFilename + TEXT("_") + MaterialName;
			}

			MaterialFilename.Append(TEXT(".png"));

			if (MaterialName == "MotionVector") {

				// auto motionVectorOutput = AddVelocityCombinePass(GraphBuilder, View, Inputs.SceneDepth.Texture, Inputs.Velocity.Texture);
				FRDGTextureRef motionVectorOutput = AddMotionVectorAndUsedCountPass(GraphBuilder, View, Inputs.SceneDepth.Texture, Inputs.Velocity.Texture);

				FScreenPassTexture MotionVectorTexture;
				MotionVectorTexture.ViewRect = View.ViewRect;
				MotionVectorTexture.Texture = motionVectorOutput;

				AddDumpToFilePass(GraphBuilder, MotionVectorTexture, MaterialFilename);

			}
			else if (MaterialName == "Matrix") {

				FString MatrixFilename = FString::Format(*BaseFilename, FormatMappings);

				// If the format made no change to the string, we add the name of the material to ensure uniqueness
				if (MatrixFilename == BaseFilename) {
					MatrixFilename = BaseFilename + TEXT("_") + MaterialName;
				}

				MatrixFilename.Append(TEXT(".txt"));

				FString output = "";
				output += "ClipToView: " + View.CachedViewUniformShaderParameters->ClipToView.ToString() + '\n';
				output += "ViewMatrix: " + View.ViewMatrices.GetViewMatrix().ToString() + '\n';
				output += "ProjectionMatrix: " + View.ViewMatrices.GetProjectionMatrix().ToString() + '\n';
				output += FString::Printf(TEXT("FOV: %g\n"), View.FOV);
				output += FString::Printf(TEXT("NearClipDistance: %g\n"), View.NearClippingDistance);
				float FarClipDistance = View.SceneViewInitOptions.OverrideFarClippingPlaneDistance > 0.0f ? View.SceneViewInitOptions.OverrideFarClippingPlaneDistance : 0;
				output += FString::Printf(TEXT("FarClipDistance: %g\n"), FarClipDistance);

				FFileHelper::SaveStringToFile(output, *MatrixFilename);
			}
			else {
				AddDumpToFilePass(GraphBuilder, Output, MaterialFilename);
			}
		}

		if (Inputs.bOverview)
		{
			FDownsamplePassInputs DownsampleInputs;
			DownsampleInputs.Name = TEXT("MaterialHalfSize");
			DownsampleInputs.SceneColor = Output;
			DownsampleInputs.Flags = EDownsampleFlags::ForceRaster;
			DownsampleInputs.Quality = EDownsampleQuality::Low;

			FScreenPassTexture HalfSize = AddDownsamplePass(GraphBuilder, View, DownsampleInputs);

			DownsampleInputs.Name = TEXT("MaterialQuarterSize");
			DownsampleInputs.SceneColor = HalfSize;

			FVisualizeBufferTile Tile;
			Tile.Input = AddDownsamplePass(GraphBuilder, View, DownsampleInputs);
			Tile.Label = GetBufferVisualizationData().GetMaterialDisplayName(FName(*MaterialName)).ToString();
#if !(UE_BUILD_SHIPPING || UE_BUILD_TEST)
			Tile.bSelected = 
				PostProcessSettings.bBufferVisualizationOverviewTargetIsSelected &&
				PostProcessSettings.BufferVisualizationOverviewSelectedTargetMaterialName == MaterialName;
#endif
			Tiles.Add(Tile);
		}
	}

	if (Inputs.bOverview)
	{
		FVisualizeBufferInputs PassInputs;
		PassInputs.OverrideOutput = Inputs.OverrideOutput;
		PassInputs.SceneColor = Inputs.SceneColor;
		PassInputs.Tiles = Tiles;

		return AddVisualizeBufferPass(GraphBuilder, View, PassInputs);
	}
	else
	{
		return Inputs.SceneColor;
	}
}
```
7. Modify Engine/Source/Runtime/Renderer/Private/PostProcess/PostProcessing.cpp:
```
if (PassSequence.IsEnabled(EPass::VisualizeGBufferOverview))
	{
		FVisualizeGBufferOverviewInputs PassInputs;
		PassSequence.AcceptOverrideIfLastPass(EPass::VisualizeGBufferOverview, PassInputs.OverrideOutput);
		PassInputs.SceneColor = SceneColor;
		PassInputs.SceneColorBeforeTonemap = SceneColorBeforeTonemap;
		PassInputs.SceneColorAfterTonemap = SceneColorAfterTonemap;
		PassInputs.SeparateTranslucency = SeparateTranslucency;
		PassInputs.Velocity = Velocity;
		PassInputs.bOverview = bVisualizeGBufferOverview;
		PassInputs.bDumpToFile = bVisualizeGBufferDumpToFile;
		PassInputs.bOutputInHDR = bOutputInHDR;

		PassInputs.SceneDepth = SceneDepth;

		SceneColor = AddVisualizeGBufferOverviewPass(GraphBuilder, View, PassInputs);
	}
```
8. Modify Engine/Source/Runtime/Renderer/Private/PostProcess/PostProcessVisualizeBuffer.h
```
struct FVisualizeGBufferOverviewInputs
{
	FScreenPassRenderTarget OverrideOutput;

	// The current scene color being processed.
	FScreenPassTexture SceneColor;
	FScreenPassTexture SceneDepth;

	// The HDR scene color immediately before tonemapping is applied.
	FScreenPassTexture SceneColorBeforeTonemap;

	// The scene color immediately after tonemapping is applied.
	FScreenPassTexture SceneColorAfterTonemap;

	// The separate translucency texture to composite.
	FScreenPassTexture SeparateTranslucency;

	// The original scene velocity texture to composite.
	FScreenPassTexture Velocity;

	// Dump targets to files on disk.
	bool bDumpToFile = false;

	// Render an overview of the GBuffer targets.
	bool bOverview = false;

	// Whether to emit outputs in HDR.
	bool bOutputInHDR = false;
};
```
