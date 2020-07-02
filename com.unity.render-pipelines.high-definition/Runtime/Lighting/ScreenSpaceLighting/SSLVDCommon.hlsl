#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"
#include "Packages/com.unity.render-pipelines.high-definition/Runtime/Material/Builtin/BuiltinData.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Packing.hlsl"

CBUFFER_START(SSLVDUniformBuffer)
float4 _AOBufferSize;
float4 _AOParams0;
float4 _AOParams1;
float4 _AOParams2;
float4 _AOParams3;
float4 _AOParams4;
float4 _FirstTwoDepthMipOffsets;
float4 _AODepthToViewParams;
CBUFFER_END

#define _AOBaseResMip  (int)_AOParams0.x
#define _AOFOVCorrection _AOParams0.y
#define _AORadius _AOParams0.z
#define _AOStepCount (uint)_AOParams0.w
#define _AOIntensity _AOParams1.x
#define _AOInvRadiusSq _AOParams1.y
#define _AOTemporalOffsetIdx _AOParams1.z
#define _AOTemporalRotationIdx _AOParams1.w
#define _AOInvStepCountPlusOne _AOParams2.z
#define _AOMaxRadiusInPixels (int)_AOParams2.w
#define _AOHistorySize _AOParams2.xy
#define _AODirectionCount _AOParams4.x
#define _FirstDepthMipOffset _FirstTwoDepthMipOffsets.xy
#define _SecondDepthMipOffset _FirstTwoDepthMipOffsets.zw

// For denoising, whether temporal or not
#define _BlurTolerance _AOParams3.x
#define _UpsampleTolerance _AOParams3.y
#define _NoiseFilterStrength _AOParams3.z
#define _StepSize _AOParams3.w
#define _AOTemporalUpperNudgeLimit _AOParams4.y
#define _AOTemporalLowerNudgeLimit _AOParams4.z
#define _AOInvHalfTanFOV _AOParams4.w

float PackAODepth(float ao, float depth)
{
    uint aoQuantized = round(saturate(ao) * 255.0);
    uint packedData = (asuint(depth) & ~0xff) | aoQuantized;
    return asfloat(packedData);
}

void UnpackAODepth(float4 packedData, out float4 ao, out float4 depths)
{
    uint4 aoQuantized = asuint(packedData) & 0xff;
    ao = aoQuantized / 255.0;
    depths = packedData;// asfloat(asuint(packedData) & ~0xff);
}

float RcpEyeDepth(float depth, float4 zBufferParam)
{
    return mad(depth, zBufferParam.z, zBufferParam.w);
}

//float4 GatherDepths(float2 localUVs)
//{
//    // this comment exists to invalidate unity's broken shader cache
//    return GATHER_TEXTURE2D_X(_CameraDepthTexture, s_linear_clamp_sampler, localUVs*float2(1., 2./3));
//}

float GetMinDepth(float2 localUVs)
{
    localUVs.x = localUVs.x * 0.5f;
    localUVs.y = localUVs.y * (1.0f / 3.0f) + (2.0f / 3.0f);
    float4 gatheredDepth = GATHER_TEXTURE2D_X(_CameraDepthTexture, s_linear_clamp_sampler, localUVs);
    return min(Min3(gatheredDepth.x, gatheredDepth.y, gatheredDepth.z), gatheredDepth.w);
}

float GetDepthForCentral(float2 positionSS)
{
    return LOAD_TEXTURE2D_X(_CameraDepthTexture, positionSS).r;
}

float GetDepthSample(float2 positionSS, uint mipLevel)
{
    if (mipLevel == 1)
    {
        positionSS = positionSS / 2 + _FirstDepthMipOffset;
    }
    else if (mipLevel == 2)
    {
        positionSS = positionSS / 4 + _SecondDepthMipOffset;
    }
    return LOAD_TEXTURE2D_X(_CameraDepthTexture, positionSS).r;
}

void NormalizeRange(inout float s0, inout float4 s1, inout float4 s5)
{
    float2 c = float2(rsqrt(3 * PI * PI * PI / 32), rsqrt(5 * PI / 9)); // second factor isn't quite right, but it's close enough - the convervatively correct number would be too conservative for most cases
    s0 *= rsqrt(4 * PI);
    s1 *= c.xxxy;
    s5 *= c.yyyy;
}

void RestoreRange(inout float s0, inout float4 s1, inout float4 s5)
{
    float2 c = float2(sqrt(3 * PI * PI * PI / 32), sqrt(5 * PI / 9));
    s0 *= sqrt(4 * PI);
    s1 *= c.xxxy;
    s5 *= c.yyyy;
}

uint Float4SNormToUInt(float4 v)
{
    //uint4 tmp = uint4(round(clamp(v, -1, +1) * 127.) * float4(1 << 0, 1 << 8, 1 << 16, 1 << 24));
    uint4 tmp = uint4(round(saturate(mad(v, 0.5, 0.5))*254.) * float4(1 << 0, 1 << 8, 1 << 16, 1 << 24));
    return tmp.x | tmp.y | tmp.z | tmp.w;
}

float4 UIntToFloat4SNorm(uint v)
{
    float4 result = uint4(0xff, 0xff00, 0xff0000, 0xff000000) & v;
    result = mad(result, rcp(float4(127 << 0, 127 << 8, 127 << 16, 127 << 24)), -1.);
    return result;
}

float4 UIntToFloat4SNormWeightedSum(uint4 v, float4 weights)
{
    return
        mad(UIntToFloat4SNorm(v[3]), weights[3],
            mad(UIntToFloat4SNorm(v[2]), weights[2],
                mad(UIntToFloat4SNorm(v[1]), weights[1], UIntToFloat4SNorm(v[0])*weights[0])));
}

float OutputFinalAO(float AO)
{
    return 1.0f - AO;// PositivePow(AO, _AOIntensity);
}
