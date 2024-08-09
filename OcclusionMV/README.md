# Occlusion Motion Vector

## How to get forward mv with backward mv
Assuming the pixel $x_{i}$ at frame i and the corresponding pixel $x_{i-1}$ at frame i-1. To get the occlusion mv at frame i, we need the backward mv from i to i-1 and forward mv from i-1 to i. These vectors are basically the same in both length and direction. The only difference is **where** this vector is written to. For backward mv, it is written to pixel $x_{i}$, and for forward mv, it is at pixel $x_{i-1}$.

To get forward mv, we can read the backward mv at pixel $x_{i}$, finding the pixel $x_{i-1}$ by: $$x_{i-1} = x_{i} - {mv}_{i}, $$ and write this vector to pixel $x_{i-1}$. This operation is usually called **splatting**. We also implement occlusion mv with this method in our data processing script. 

When the pixel $x_{i-1}$ is splatted to frame i-1, there are two problems need to be solved:
1. $x_{i-1}$ might not be situated at the center of pixel; 
2. There might be more than one pixel from frame i that are splatted to pixel $x_{i-1}$.  

For the first problem, we simply write the mv to the neighbor 4 pixels that surround pixel $x_{i-1}$ ( another option is to write to the nearest pixel, like the method used in "Mob-FGSR: Frame Generation and Super Resolution for Mobile
Real-Time Rendering"). For the second one, we use atomic operation in shader to make the **closer** pixel write to forward mv buffer.

## Implementation
For occlusion mv warping, it is not necessary to write occlusion mv explicitly, as occlusion mv can be computed from backward mv and forward mv when operating warping. 

The example shaders used to compute forward mv is shown in ForwardMVSplatDepthComputeShader.usf and ForwardMVSplatComputeShader.usf. 
We use two passes to compute forwawrd mv:
1. Splatting depth from frame i to frame i-1 with backward mv, using InterlockedMin to get closest depth;
2. Splatting mv from frame i to frame i-1 with backward mv, using NvInterlockedExchangeUint64.

There is another implementation that only needs single splatting operation:
1. Packing 32bit motion vector (R16G16) and 32 bit depth to a 64bit uint;
2. Splatting this uint value with NvInterlockedMinUint64.

Please refer to [Rendering Point Clouds with Compute Shaders and
Vertex Order Optimization](https://arxiv.org/abs/2104.07526) if you are interested in this implementation.

