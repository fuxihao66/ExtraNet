# How to use higher version of Unreal Engine to export data

If you want to use scenes that only support higher version of Unreal Engine (like 4.27), you can try to follow the instructions below to modify the engine by yourselves.

0. Clone the official version of UE from github.
1. Move ".ush" and ".usf" files to the Engine/Shaders/private folder
2. Move ".cpp" and ".h" files to Engine/Source/Runtime/Renderer/Private/PostProcess folder
3. Run the **Setup** and **GenerateProject** scripts
4. Move ".uasset" files in ExtraNet/EngineModificationGuide/BufferVisualization to Engine/Content/BufferVisualization folder
5. Add the following lines to BaseEngine.ini
```
WorldPosition=(Material="/Engine/BufferVisualization/WorldPosition.WorldPosition", Name=LOCTEXT("WorldPosition", "World Position"))
MyStencil=(Material="/Engine/BufferVisualization/MyStencil.MyStencil", Name=LOCTEXT("MyStencil", "Custom Stencil Value"))
NoV=(Material="/Engine/BufferVisualization/NoV.NoV", Name=LOCTEXT("NoV", "NoV"))
MotionVector=(Material="/Engine/BufferVisualization/MotionVector.MotionVector", Name=LOCTEXT("MotionVector", "MotionVector"))
```

