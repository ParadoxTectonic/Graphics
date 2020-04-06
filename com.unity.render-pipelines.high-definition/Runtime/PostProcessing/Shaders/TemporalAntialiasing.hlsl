#define HDR_MAPUNMAP        1
#define RADIUS              0.75
#define FEEDBACK_MIN        0.94
#define FEEDBACK_MAX        0.9
#define COLOR_DEVIATION_ALLOWED_MIN 0.3
#define COLOR_DEVIATION_ALLOWED_MAX 1.25
#define SHARPEN             1

#define CLAMP_MAX       65472.0 // HALF_MAX minus one (2 - 2^-9) * 2^15

#if !defined(CTYPE)
    #define CTYPE float3
#endif

#if UNITY_REVERSED_Z
    #define IS_NEARER_Z(a, b) ((a) > (b))
#else
    #define IS_NEARER_Z(a, b) ((a) < (b))
#endif

////////////////////////////////////

float4 SampleWeighted(TEXTURE2D_X(tex), float2 uv, float w, in out float wsum)
{
    wsum += w;
    //return tex.SampleLevel(LinearSampler, uv, 0)*w;
    return SAMPLE_TEXTURE2D_X_LOD(tex, s_linear_clamp_sampler, uv, 0)*w;

}

float2 EvalLanczos3t3(float2 f)
{
    return 0.5 + (-1.46217 + 2.92433 * f) * rcp(2.92408 + f * (-1. + f));
}

float2 EvalLanczos3w3(float2 f)
{
    return 0.998157 + (0.866948 - 0.866948 * f) * f;
}

float2 EvalLanczos3w2(float2 f)
{
    return 2.1734 - 1.08916 * f + (-4.09402 + 1.45537 * f) * rcp(1.88435 + f * (-0.451784 + f));
}

float2 EvalLanczos3w1(float2 f)
{
    return -0.432408 + 0.269292 * f + (0.532468 - 0.227442 * f) * rcp(1.23146 + f * (-0.362435 + f));
}

float4 FetchLanczos(TEXTURE2D_X(tex), float2 coords, float2 offset, float2 scale)
{
    float2 pixelLoc = (coords*_ScreenSize.xy + offset)*scale;
    float2 rcpTextureSize = _ScreenSize.zw;

    float2 f = frac(pixelLoc - 0.5);

    float2 t1 = 0.;
    float2 t2 = 0.;
    float2 t3 = EvalLanczos3t3(f);
    float2 t4 = 1.;
    float2 t5 = 1.;

    float2 w1 = EvalLanczos3w1(f);
    float2 w2 = EvalLanczos3w2(f);
    float2 w3 = EvalLanczos3w3(f);
    float2 w4 = EvalLanczos3w2(1. - f);
    float2 w5 = EvalLanczos3w1(1. - f);

    float2 p1 = (-f + t1 - 2. + pixelLoc)*rcpTextureSize;
    float2 p2 = (-f + t2 - 1. + pixelLoc)*rcpTextureSize;
    float2 p3 = (-f + t3 + 0. + pixelLoc)*rcpTextureSize;
    float2 p4 = (-f + t4 + 1. + pixelLoc)*rcpTextureSize;
    float2 p5 = (-f + t5 + 2. + pixelLoc)*rcpTextureSize;

    float wsum = 0.;
    float4 result = 0.;
    //result += SampleWeighted(tex, float2(p1.x, p1.y), w1.x*w1.y, wsum);
    //result += SampleWeighted(tex, float2(p2.x, p1.y), w2.x*w1.y, wsum);
    result += SampleWeighted(tex, float2(p3.x, p1.y), w3.x*w1.y, wsum);
    //result += SampleWeighted(tex, float2(p4.x, p1.y), w4.x*w1.y, wsum);
    //result += SampleWeighted(tex, float2(p5.x, p1.y), w5.x*w1.y, wsum);

    //result += SampleWeighted(tex, float2(p1.x, p2.y), w1.x*w2.y, wsum);
    result += SampleWeighted(tex, float2(p2.x, p2.y), w2.x*w2.y, wsum);
    result += SampleWeighted(tex, float2(p3.x, p2.y), w3.x*w2.y, wsum);
    result += SampleWeighted(tex, float2(p4.x, p2.y), w4.x*w2.y, wsum);
    //result += SampleWeighted(tex, float2(p5.x, p2.y), w5.x*w2.y, wsum);

    result += SampleWeighted(tex, float2(p1.x, p3.y), w1.x*w3.y, wsum);
    result += SampleWeighted(tex, float2(p2.x, p3.y), w2.x*w3.y, wsum);
    result += SampleWeighted(tex, float2(p3.x, p3.y), w3.x*w3.y, wsum);
    result += SampleWeighted(tex, float2(p4.x, p3.y), w4.x*w3.y, wsum);
    result += SampleWeighted(tex, float2(p5.x, p3.y), w5.x*w3.y, wsum);

    //result += SampleWeighted(tex, float2(p1.x, p4.y), w1.x*w4.y, wsum);
    result += SampleWeighted(tex, float2(p2.x, p4.y), w2.x*w4.y, wsum);
    result += SampleWeighted(tex, float2(p3.x, p4.y), w3.x*w4.y, wsum);
    result += SampleWeighted(tex, float2(p4.x, p4.y), w4.x*w4.y, wsum);
    //result += SampleWeighted(tex, float2(p5.x, p4.y), w5.x*w4.y, wsum);

    //result += SampleWeighted(tex, float2(p1.x, p5.y), w1.x*w5.y, wsum);
    //result += SampleWeighted(tex, float2(p2.x, p5.y), w2.x*w5.y, wsum);
    result += SampleWeighted(tex, float2(p3.x, p5.y), w3.x*w5.y, wsum);
    //result += SampleWeighted(tex, float2(p4.x, p5.y), w4.x*w5.y, wsum);
    //result += SampleWeighted(tex, float2(p5.x, p5.y), w5.x*w5.y, wsum);

    return result*rcp(wsum);
}

////////////////////////////////////

float3 Fetch(TEXTURE2D_X(tex), float2 coords, float2 offset, float2 scale)
{
    float2 uv = (coords + offset * _ScreenSize.zw) * scale;
    return SAMPLE_TEXTURE2D_X_LOD(tex, s_linear_clamp_sampler, uv, 0).xyz;
}

float2 Fetch2(TEXTURE2D_X(tex), float2 coords, float2 offset, float2 scale)
{
    float2 uv = (coords + offset * _ScreenSize.zw) * scale;
    return SAMPLE_TEXTURE2D_X_LOD(tex, s_linear_clamp_sampler, uv, 0).xy;
}


float4 Fetch4(TEXTURE2D_X(tex), float2 coords, float2 offset, float2 scale)
{
    float2 uv = (coords + offset * _ScreenSize.zw) * scale;
    return SAMPLE_TEXTURE2D_X_LOD(tex, s_linear_clamp_sampler, uv, 0);
}

float4 Fetch4Array(Texture2DArray tex, uint slot, float2 coords, float2 offset, float2 scale)
{
    float2 uv = (coords + offset * _ScreenSize.zw) * scale;
    return SAMPLE_TEXTURE2D_ARRAY_LOD(tex, s_linear_clamp_sampler, uv, slot, 0);
}

float3 Map(float3 x)
{
    #if HDR_MAPUNMAP
    return FastTonemap(x);
    #else
    return x;
    #endif
}

float3 Unmap(float3 x)
{
    #if HDR_MAPUNMAP
    return FastTonemapInvert(x);
    #else
    return x;
    #endif
}

float MapPerChannel(float x)
{
    #if HDR_MAPUNMAP
    return FastTonemapPerChannel(x);
    #else
    return x;
    #endif
}

float UnmapPerChannel(float x)
{
    #if HDR_MAPUNMAP
    return FastTonemapPerChannelInvert(x);
    #else
    return x;
    #endif
}

float2 MapPerChannel(float2 x)
{
    #if HDR_MAPUNMAP
    return FastTonemapPerChannel(x);
    #else
    return x;
    #endif
}

float2 UnmapPerChannel(float2 x)
{
    #if HDR_MAPUNMAP
    return FastTonemapPerChannelInvert(x);
    #else
    return x;
    #endif
}

float2 GetClosestFragment(float2 positionSS)
{
    float center  = LoadCameraDepth(positionSS);
    float nw = LoadCameraDepth(positionSS + float2(-1, -1));
    float ne = LoadCameraDepth(positionSS + float2( 1, -1));
    float sw = LoadCameraDepth(positionSS + float2(-1,  1));
    float se = LoadCameraDepth(positionSS + float2( 1,  1));

    float3 closest = float3(0.0, 0.0, center);
    closest = IS_NEARER_Z(nw, closest.z) ? float3(-1, -1, nw) : closest;
    closest = IS_NEARER_Z(ne, closest.z) ? float3( 1, -1, ne) : closest;
    closest = IS_NEARER_Z(sw, closest.z) ? float3(-1,  1, sw) : closest;
    closest = IS_NEARER_Z(se, closest.z) ? float3( 1,  1, se) : closest;

    return positionSS + closest.xy;
}

CTYPE ClipToAABB(CTYPE color, CTYPE minimum, CTYPE maximum)
{
    // note: only clips towards aabb center (but fast!)
    CTYPE center  = 0.5 * (maximum + minimum);
    CTYPE extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    CTYPE offset = color - center;
    
    CTYPE ts = abs(extents) / max(abs(offset), 1e-4);
    float t = saturate(Min3(ts.x, ts.y,  ts.z));
    return center + offset * t;
}
