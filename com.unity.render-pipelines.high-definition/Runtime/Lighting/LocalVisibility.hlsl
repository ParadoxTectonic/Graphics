#ifndef LOCALVISIBILITY_HLSL
#define LOCALVISIBILITY_HLSL

struct LocalVisibility
{
    real harmonics0;        // DC term
    real4 harmonics1;       // 3 linear terms + 1 quadratic
    real4 harmonics5;       // 4 quadratic terms
};

LocalVisibility FullLocalVisibility()
{
    LocalVisibility result;
    result.harmonics0 = sqrt(4 * PI);
    result.harmonics1 = 0;// float4(0.0001, 0, 0, 0);   // nonzero so dominant direction is established
    result.harmonics5 = 0;
    return result;
}

float Ramp(float lo, float hi, float x)
{
    return saturate((x - lo) / (hi - lo));
}

LocalVisibility LocalVisibilityHemisphereUnnormalized(float3 n)
{
    LocalVisibility result;
    // Rotated zonal harmonic for max(0, cos(theta))^(1/8) - a step-like function for the positive hemisphere
    result.harmonics0 = 1.57551;
    result.harmonics1.xyz = 1.35441 * n.yzx;    // this coefficient has been tuned from 1.4447 to eliminate some hard cutoff
    result.harmonics1[3] = 0.244078 * n.x*n.y;
    result.harmonics5[0] = 0.244078 * n.y*n.z;
    result.harmonics5[1] = 0.0704592 * (3*n.z*n.z - 1);
    result.harmonics5[2] = 0.244078 * n.x*n.z;
    result.harmonics5[3] = 0.122039 * (n.x*n.x - n.y*n.y);
    return result;
}

LocalVisibility LocalVisibilityNormalizedCosine(float3 n)
{
    LocalVisibility result;
    result.harmonics0 = rsqrt(4*PI);
    result.harmonics1.xyz = rsqrt(3*PI) * n.yzx;
    result.harmonics1[3] = rsqrt(64*PI/15) * n.x*n.y;
    result.harmonics5[0] = rsqrt(64*PI/15) * n.y*n.z;
    result.harmonics5[1] = rsqrt(256*PI/5) * (3*n.z*n.z - 1);
    result.harmonics5[2] = rsqrt(64*PI/15) * n.x*n.z;
    result.harmonics5[3] = rsqrt(256*PI/15) * (n.x*n.x - n.y*n.y);
    return result;
}

float LocalVisibilityInnerProduct(LocalVisibility a, LocalVisibility b)
{
    return a.harmonics0*b.harmonics0 + dot(a.harmonics1, b.harmonics1) + dot(a.harmonics5, b.harmonics5);
}

float ToeStep(float x)
{
    x = saturate(x);
    return lerp(x * x, x, x);
}

void EvaluateLocalVisibilityRect(LocalVisibility localVisibility, LocalVisibility cosWeightedVisibility, float4x3 lightVerts, float3 n, out float specularVisibility, out float diffuseVisibility)
{
    LocalVisibility surfaceHemi = LocalVisibilityHemisphereUnnormalized(n);
    LocalVisibility surfaceCos = LocalVisibilityNormalizedCosine(n);

    // The idea here is to compute the ratio of occluded visibility to unoccluded visibility.
    // For specular, unoccluded visibility starts as the full hemisphere above the surface. For diffuse, a cosine lobe.
    // Then, mask off exterior of the light quad by treating each edge as a plane (so, also a full hemisphere).
    // So we are computing the inner product of the occluded and unoccluded visibility with the interior halfspace of each edge.
    // Do this for each edge and compute the product of all 4 scalar results - now you have the interior of the quad, take 4th root to restore linearity.

    float diffuseDenominator, specularDenominator;
    LocalVisibility edgeHemi;
    edgeHemi = LocalVisibilityHemisphereUnnormalized(normalize(cross(lightVerts[0], lightVerts[1])));
    specularVisibility = max(0, LocalVisibilityInnerProduct(localVisibility, edgeHemi));
    specularDenominator = max(0, LocalVisibilityInnerProduct(surfaceHemi, edgeHemi));
    diffuseVisibility = max(0, LocalVisibilityInnerProduct(cosWeightedVisibility, edgeHemi));
    diffuseDenominator = max(0, LocalVisibilityInnerProduct(surfaceCos, edgeHemi));

    edgeHemi = LocalVisibilityHemisphereUnnormalized(normalize(cross(lightVerts[1], lightVerts[2])));
    specularVisibility *= max(0, LocalVisibilityInnerProduct(localVisibility, edgeHemi));
    specularDenominator *= max(0, LocalVisibilityInnerProduct(surfaceHemi, edgeHemi));
    diffuseVisibility *= max(0, LocalVisibilityInnerProduct(cosWeightedVisibility, edgeHemi));
    diffuseDenominator *= max(0, LocalVisibilityInnerProduct(surfaceCos, edgeHemi));

    edgeHemi = LocalVisibilityHemisphereUnnormalized(normalize(cross(lightVerts[2], lightVerts[3])));
    specularVisibility *= max(0, LocalVisibilityInnerProduct(localVisibility, edgeHemi));
    specularDenominator *= max(0, LocalVisibilityInnerProduct(surfaceHemi, edgeHemi));
    diffuseVisibility *= max(0, LocalVisibilityInnerProduct(cosWeightedVisibility, edgeHemi));
    diffuseDenominator *= max(0, LocalVisibilityInnerProduct(surfaceCos, edgeHemi));

    edgeHemi = LocalVisibilityHemisphereUnnormalized(normalize(cross(lightVerts[3], lightVerts[0])));
    specularVisibility *= max(0, LocalVisibilityInnerProduct(localVisibility, edgeHemi));
    specularDenominator *= max(0, LocalVisibilityInnerProduct(surfaceHemi, edgeHemi));
    diffuseVisibility *= max(0, LocalVisibilityInnerProduct(cosWeightedVisibility, edgeHemi));
    diffuseDenominator *= max(0, LocalVisibilityInnerProduct(surfaceCos, edgeHemi));

    specularVisibility = ToeStep(rsqrt(rsqrt(specularVisibility / (specularDenominator + 1e-5f))));
    diffuseVisibility = ToeStep(rsqrt(rsqrt(diffuseVisibility / (diffuseDenominator + 1e-5f))));
}

float EvaluateLocalVisibilityCosine(LocalVisibility localVisibility, float3 dirWS)
{
    float result = localVisibility.harmonics0 * rsqrt(PI * 4);
    result += dot(localVisibility.harmonics1.xyz, dirWS.yzx) * (rsqrt(PI * 4 / 3) * 2./3.);
    result += localVisibility.harmonics1.w * dirWS.x * dirWS.y * (rsqrt(PI * 4 / 15) * 0.25);
    result += dot(localVisibility.harmonics5.xz, dirWS.yx * dirWS.zz) * (rsqrt(PI * 4 / 15) * 0.25);
    result += localVisibility.harmonics5[1] * mad(dirWS.z * dirWS.z, 3, -1) * (rsqrt(PI * 16 / 5) * 0.25);
    result += localVisibility.harmonics5[3] * (dirWS.x * dirWS.x - dirWS.y * dirWS.y) * (rsqrt(PI * 16 / 15) * 0.25);
    return smoothstep(0., 1., result);
    return max(result, 0);
}

float EvaluateLocalVisibilityDirac(LocalVisibility localVisibility, float3 dirWS)
{
    float result = localVisibility.harmonics0 * rsqrt(PI * 4);
    result += dot(localVisibility.harmonics1.xyz, dirWS.yzx) * rsqrt(PI * 4 / 3);
    result += 0.4 * localVisibility.harmonics1.w * dirWS.x * dirWS.y * rsqrt(PI * 4 / 15);
    result += 0.4 * dot(localVisibility.harmonics5.xz, dirWS.yx * dirWS.zz) * rsqrt(PI * 4 / 15);
    result += 0.4 * localVisibility.harmonics5[1] * mad(dirWS.z * dirWS.z, 3, -1) * rsqrt(PI * 16 / 5);
    result += 0.4 * localVisibility.harmonics5[3] * (dirWS.x * dirWS.x - dirWS.y * dirWS.y) * rsqrt(PI * 16 / 15);
    return smoothstep(0., 1., result);
    return max(result, 0);
}

float EvaluateLocalVisibilityDiracSharp(LocalVisibility localVisibility, float3 dirWS)
{
    float result = localVisibility.harmonics0 * rsqrt(PI * 4);
    result += dot(localVisibility.harmonics1.xyz, dirWS.yzx) * rsqrt(PI * 4 / 3);
    result += 0.4 * localVisibility.harmonics1.w * dirWS.x * dirWS.y * rsqrt(PI * 4 / 15);
    result += 0.4 * dot(localVisibility.harmonics5.xz, dirWS.yx * dirWS.zz) * rsqrt(PI * 4 / 15);
    result += 0.4 * localVisibility.harmonics5[1] * mad(dirWS.z * dirWS.z, 3, -1) * rsqrt(PI * 16 / 5);
    result += 0.4 * localVisibility.harmonics5[3] * (dirWS.x * dirWS.x - dirWS.y * dirWS.y) * rsqrt(PI * 16 / 15);
    return saturate(1.40841 * exp2(-0.434783 * rcp(max(result - 0.12, 0))));
    return sqrt(Ramp(0.2, 1., result));
    return smoothstep(0.25, 0.75, result);
    return max(result, 0);
}

float EstimateSpecularVisibilityCorrection(LocalVisibility localVisibility, float3 dirWS, float perceptualRoughness)
{
#if 1
    return saturate(EvaluateLocalVisibilityDirac(localVisibility, dirWS));
#else
    float cosHorizon = mad(localVisibility.harmonics0, -rsqrt(PI), 1);
    float3 axis = normalize(localVisibility.harmonics1.zxy + float3(0, 0.0001, 0));
    return Ramp(-perceptualRoughness, perceptualRoughness, dot(dirWS, axis) - cosHorizon);
#endif
}

float3x3 GetLocalFrameFast(float3 n)
{
    float3 v = n.yzx;
    v.x = -v.x;
    v = normalize(cross(v, n));
    return float3x3(cross(n, v), v, n);
}

void RotateSphericalHarmonicsL2(float3x3 m, float s0, float4 s1, float4 s5, out float r0, out float4 r1, out float4 r5)
{
    // L0
    r0 = s0;

    // L1
    r1.zxy = mul(m, s1.zxy);

    // L2
    float2 s4 = float2(s5[0], s1[3]);
    s4 += s5[3] - s5[2];
    s4[0] += s5[3];
    s4[1] += sqrt(float(3)) * s5[1];
    float3 s6 = float3(s1[3], s5.zx);

    float3x3 b = float3x3(m[0].xxy + m[0].yzz, m[1].xxy + m[1].yzz, m[2].xxy + m[2].yzz);

    r1[3] = dot(s4.xy, m[0].xz * m[1].xz) + dot(s6, b[0] * b[1]);
    r5[0] = dot(s4.xy, m[1].xz * m[2].xz) + dot(s6, b[1] * b[2]);
    r5[1] = float(sqrt(3.) / 2) * (dot(s4.xy, mad(m[2].xz, m[2].xz, float(-1. / 3))) + dot(s6, mad(b[2], b[2], float(-2. / 3))));
    r5[2] = dot(s4.xy, m[0].xz * m[2].xz) + dot(s6, b[0] * b[2]);
    r5[3] = float(0.5) * (dot(s4.xy, m[0].xz * m[0].xz - m[1].xz * m[1].xz) + dot(s6, b[0] * b[0] - b[1] * b[1]));
}

LocalVisibility RotateLocalVisibility(float3x3 m, LocalVisibility s)
{
    LocalVisibility r;
    RotateSphericalHarmonicsL2(m, s.harmonics0, s.harmonics1, s.harmonics5, r.harmonics0, r.harmonics1, r.harmonics5);
    return r;
}

void ModulateClampedCosineZH2(float s0, float4 s1, float4 s5, out float r0, out float4 r1, out float4 r5)
{
    r0 = s0 * (0.25 / PI) + s1[1] * (rsqrt(12.) / PI) + s5[1] * (sqrt(5.) / (16 * PI));
    r1[0] = s1[0] * (3. / (16 * PI)) + s5[0] * (rsqrt(20.) / PI);
    r1[1] = s1[1] * (3. / (8 * PI)) + s0 * (rsqrt(12.) / PI) + s5[1] * (rsqrt(15.) / PI);
    r1[2] = s1[2] * (3. / (16 * PI)) + s5[2] * (rsqrt(20.) / PI);
    r1[3] = s1[3] * (5. / (32 * PI));
    r5[0] = s5[0] * (5. / (16 * PI)) + s1[0] * (rsqrt(20.) / PI);
    r5[1] = s5[1] * (5. / (16 * PI)) + s0 * (sqrt(5.) / (16 * PI)) + s1[1] * (rsqrt(15.) / PI);
    r5[2] = s5[2] * (5. / (16 * PI)) + s1[2] * (rsqrt(20.) / PI);
    r5[3] = s5[3] * (5. / (32 * PI));
}

void ModulateClampedHemisphereZH2(float s0, float4 s1, float4 s5, out float r0, out float4 r1, out float4 r5)
{
    r0 = s0 * 0.5 + s1[1] * 0.433013;
    r1[0] = s1[0] * 0.5 + s5[0] * 0.419263;
    r1[1] = s1[1] * 0.5 + s0 * 0.433013 + s5[1] * 0.242061;
    r1[2] = s1[2] * 0.5 + s5[2] * 0.419263;
    r1[3] = s1[3] * 0.5;
    r5[0] = s5[0] * 0.5 + s1[0] * 0.419263;
    r5[1] = s5[1] * 0.5 + s1[1] * 0.242061;
    r5[2] = s5[2] * 0.5 + s1[2] * 0.419263;
    r5[3] = s5[3] * 0.5;
}

LocalVisibility ModulateClampedCosineZH(LocalVisibility s)
{
    LocalVisibility r;
    ModulateClampedCosineZH2(s.harmonics0, s.harmonics1, s.harmonics5, r.harmonics0, r.harmonics1, r.harmonics5);
    return r;
}

LocalVisibility ModulateClampedHemisphereZH(LocalVisibility s)
{
    LocalVisibility r;
    ModulateClampedHemisphereZH2(s.harmonics0, s.harmonics1, s.harmonics5, r.harmonics0, r.harmonics1, r.harmonics5);
    return r;
}

LocalVisibility ModulateLocalVisibilityWithClampedCosine(LocalVisibility localVisibility, float3 n)
{
    LocalVisibility result;
    float3x3 worldToNormal = GetLocalFrameFast(n);
    RotateSphericalHarmonicsL2(worldToNormal, localVisibility.harmonics0, localVisibility.harmonics1, localVisibility.harmonics5, result.harmonics0, result.harmonics1, result.harmonics5);
    ModulateClampedCosineZH2(result.harmonics0, result.harmonics1, result.harmonics5, result.harmonics0, result.harmonics1, result.harmonics5);
    RotateSphericalHarmonicsL2(transpose(worldToNormal), result.harmonics0, result.harmonics1, result.harmonics5, result.harmonics0, result.harmonics1, result.harmonics5);
    return result;
}

LocalVisibility ModulateLocalVisibilityWithHemisphere(LocalVisibility localVisibility, float3 n)
{
    LocalVisibility result;
    float3x3 worldToNormal = GetLocalFrameFast(n);
    RotateSphericalHarmonicsL2(worldToNormal, localVisibility.harmonics0, localVisibility.harmonics1, localVisibility.harmonics5, result.harmonics0, result.harmonics1, result.harmonics5);
    ModulateClampedHemisphereZH2(result.harmonics0, result.harmonics1, result.harmonics5, result.harmonics0, result.harmonics1, result.harmonics5);
    RotateSphericalHarmonicsL2(transpose(worldToNormal), result.harmonics0, result.harmonics1, result.harmonics5, result.harmonics0, result.harmonics1, result.harmonics5);
    return result;
}

#endif