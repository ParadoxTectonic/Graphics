Shader "Hidden/HDRP/TemporalAntialiasing"
{
    Properties
    {
        [HideInInspector] _StencilRef("_StencilRef", Int) = 2
        [HideInInspector] _StencilMask("_StencilMask", Int) = 2
    }

    HLSLINCLUDE

        #pragma target 4.5
        #pragma multi_compile_local _ ORTHOGRAPHIC
        #pragma multi_compile_local _ REDUCED_HISTORY_CONTRIB
        #pragma multi_compile_local _ ENABLE_ALPHA
        #pragma only_renderers d3d11 playstation xboxone vulkan metal switch

        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Material/Builtin/BuiltinData.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/PostProcessDefines.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/TemporalAntialiasing.hlsl"

        TEXTURE2D_X(_InputTexture);
        TEXTURE2D_X(_InputHistoryTexture);
    #if defined(SHADER_API_GAMECORE)
        RW_TEXTURE2D_X(CTYPE, _OutputHistoryTexture) : register(u1);
    #else
        RW_TEXTURE2D_X(CTYPE, _OutputHistoryTexture);
    #endif

        struct Attributes
        {
            uint vertexID : SV_VertexID;
            UNITY_VERTEX_INPUT_INSTANCE_ID
        };

        struct Varyings
        {
            float4 positionCS : SV_POSITION;
            float2 texcoord   : TEXCOORD0;
            UNITY_VERTEX_OUTPUT_STEREO
        };

        Varyings Vert(Attributes input)
        {
            Varyings output;
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
            output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
            output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);
            return output;
        }

        float3 ClipToCylinder(float3 v, float3 halfExtent)
        {
            v *= rcp(halfExtent);
            float3 v2 = v*v;
            v *= halfExtent*rsqrt(Max3(1., v2.x, v2.y + v2.z));
            //v *= halfExtent*rsqrt(max(1., float3(v2.x, v2.yy + v2.zz)));
            return v;
        }

        float Hmax(float3 v)
        {
            return Max3(v.x, v.y, v.z);
        }

        float3 ClipToBox(float3 v, float3 halfExtent)
        {
            v *= rcp(halfExtent);
            v *= halfExtent * rcp(max(1., Hmax(abs(v))));
            return v;
        }

        float3 MapColor(float3 x)
        {
        #if HDR_MAPUNMAP
            float3 y = mul(float3x3(0.25, 0.5, 0.25, -0.25, 0.5, -0.25, 0.5, 0, -0.5), x);  // RGB -> YCoCg
            y.rgb *= rcp(0.5 + y.r);
            return y;
        #else
            return x;
        #endif
        }

        float3 UnmapColor(float3 x)
        {
        #if HDR_MAPUNMAP
            x.rgb *= 0.5*rcp(1. - x.r);
            return mul(float3x3(1., -1., 1., 1., 1., 0., 1., -1., -1.), x);  // YCoCg -> RGB
        #else
            return x;
        #endif
        }

        float NeighborhoodStatisticsWeight(int i, int j)
        {
            static const float cWeights[] = { 16./196., 24./196., 36./196. };
            return cWeights[(i == 0) + (j == 0)];
        }

        void GatherNeighborhoodStatistics(TEXTURE2D_X(tex), float2 positionSS, out CTYPE mean, out CTYPE sigma)
        {
            mean = 0.;
            sigma = 0.;
            [unroll] for (int i = -1; i <= +1; i++)
            {
                [unroll] for (int j = -1; j <= +1; j++)
                {
                    CTYPE c = LOAD_TEXTURE2D_X(tex, positionSS + int2(i, j)).CTYPE_SWIZZLE;
                    c.rgb = MapColor(c.rgb);
                    float w = NeighborhoodStatisticsWeight(i, j);
                    mean += c*w;
                    sigma += c*c*w;
                }
            }
            sigma = sqrt(max(sigma - mean*mean, 0.000001));
        }

        bool2 IsInUnitBox(float2 v)
        {
            return v == saturate(v);
        }

        CTYPE SmoothClamp(CTYPE x, CTYPE a, CTYPE b)
        {
            return lerp(a, b, smoothstep(a, b, x));
        }

        void FragTAA(Varyings input, out CTYPE outColor : SV_Target0)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 jitter = _TaaJitterStrength.zw;

    #if defined(ORTHOGRAPHIC)
            // Don't dilate in ortho
            float2 closest = input.positionCS.xy;
    #else
            float2 closest = GetClosestFragment(input.positionCS.xy);
    #endif

            float2 motionVector;
            DecodeMotionVector(LOAD_TEXTURE2D_X(_CameraMotionVectorsTexture, closest), motionVector);
            float motionVecLength = length(motionVector);

            float2 uv = input.texcoord - jitter;


            CTYPE color = Fetch4(_InputTexture, uv, 0.0, _RTHandleScale.xy).CTYPE_SWIZZLE;
            CTYPE history = FetchLanczos(_InputHistoryTexture, input.texcoord - motionVector, 0.0, _RTHandleScaleHistory.zw).CTYPE_SWIZZLE;
            //history.rgb = max(history.rgb, 0.);

    #if SHARPEN
            //float sharpenStrength = _TaaFrameInfo.x;

            CTYPE lowPass = 0;
            lowPass += 0.25*Fetch4(_InputTexture, uv, float2(-0.5, -0.5), _RTHandleScale.xy).CTYPE_SWIZZLE;
            lowPass += 0.25*Fetch4(_InputTexture, uv, float2(+0.5, -0.5), _RTHandleScale.xy).CTYPE_SWIZZLE;
            lowPass += 0.25*Fetch4(_InputTexture, uv, float2(-0.5, +0.5), _RTHandleScale.xy).CTYPE_SWIZZLE;
            lowPass += 0.25*Fetch4(_InputTexture, uv, float2(+0.5, +0.5), _RTHandleScale.xy).CTYPE_SWIZZLE;

            color += SmoothClamp(2.*(color - lowPass),
                -0.1*color,
                +0.1*color);
    #endif

            color.xyz = clamp(color.xyz, 0.0, CLAMP_MAX);

            color.xyz = MapColor(color.xyz);
            history.xyz = MapColor(history.xyz);

            // Clip history samples
            CTYPE mean, sigma;
            GatherNeighborhoodStatistics(_InputTexture, input.positionCS.xy, mean, sigma);
            float allowedDeviation = lerp(COLOR_DEVIATION_ALLOWED_MAX, COLOR_DEVIATION_ALLOWED_MIN, saturate(motionVecLength*100.0));
            history.rgb = mean.rgb + ClipToCylinder(history.rgb - mean.rgb, sigma.rgb*allowedDeviation);

            // Blend color & history
            // Feedback weight from unbiased luminance diff (Timothy Lottes)
            float colorLuma = color.x;
            float historyLuma = history.x;
            float diff = abs(colorLuma - historyLuma) / Max3(colorLuma, historyLuma, 0.2);
            float weight = 1.0 - diff;
            float feedback = lerp(FEEDBACK_MIN, FEEDBACK_MAX, weight * weight);

    #if defined(ENABLE_ALPHA)
            // Compute the antialiased alpha value
            color.w = lerp(color.w, history.w, feedback);
            // TAA should not overwrite pixels with zero alpha. This allows camera stacking with mixed TAA settings (bottom camera with TAA OFF and top camera with TAA ON).
            CTYPE unjitteredColor = Fetch4(_InputTexture, input.texcoord - color.w * jitter, 0.0, _RTHandleScale.xy).CTYPE_SWIZZLE;
            color.xyz = lerp(MapColor(unjitteredColor.xyz), color.xyz, color.w);
            feedback *= color.w;
    #endif
            // Don't incorporate off-screen history samples
            feedback *= all(IsInUnitBox(input.texcoord - motionVector));

            color.xyz = UnmapColor(lerp(color.xyz, history.xyz, feedback));
            color.xyz = clamp(color.xyz, 0.0, CLAMP_MAX);

            _OutputHistoryTexture[COORD_TEXTURE2D_X(input.positionCS.xy)] = color;
            outColor = color; 
        }

        void FragExcludedTAA(Varyings input, out CTYPE outColor : SV_Target0)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 jitter = _TaaJitterStrength.zw;
            float2 uv = input.texcoord - jitter;

            outColor = Fetch4(_InputTexture, uv, 0.0, _RTHandleScale.xy).CTYPE_SWIZZLE;
        }
    ENDHLSL

    SubShader
    {
        Tags{ "RenderPipeline" = "HDRenderPipeline" }

        // TAA
        Pass
        {
            Stencil
            {
                ReadMask [_StencilMask]       // ExcludeFromTAA
                Ref [_StencilRef]          // ExcludeFromTAA
                Comp NotEqual
                Pass Keep
            }

            ZWrite Off ZTest Always Blend Off Cull Off

            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragTAA
            ENDHLSL
        }

        // Excluded from TAA
        // Note: This is a straightup passthrough now, but it would be interesting instead to try to reduce history influence instead.
        Pass
        {
            Stencil
            {
                ReadMask [_StencilMask]    
                Ref     [_StencilRef]
                Comp Equal
                Pass Keep
            }

            ZWrite Off ZTest Always Blend Off Cull Off

            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragExcludedTAA
            ENDHLSL
        }
    }
    Fallback Off
}
