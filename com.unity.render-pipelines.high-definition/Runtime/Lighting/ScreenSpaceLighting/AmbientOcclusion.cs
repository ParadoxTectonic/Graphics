using System;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Serialization;

namespace UnityEngine.Rendering.HighDefinition
{
    /// <summary>
    /// A volume component that holds settings for the ambient occlusion.
    /// </summary>
    [Serializable, VolumeComponentMenu("Lighting/Ambient Occlusion")]
    public sealed class AmbientOcclusion : VolumeComponentWithQuality
    {
        public const GraphicsFormat HarmonicsFormat = GraphicsFormat.R8G8B8A8_SNorm;

        /// <summary>
        /// Enable ray traced ambient occlusion.
        /// </summary>
        public BoolParameter rayTracing = new BoolParameter(false);

        /// <summary>
        /// Enable ray traced ambient occlusion.
        /// </summary>
        public BoolParameter localVisibilityDistribution = new BoolParameter(true);

        /// <summary>
        /// Controls the strength of the ambient occlusion effect. Increase this value to produce darker areas.
        /// </summary>
        public ClampedFloatParameter intensity = new ClampedFloatParameter(0f, 0f, 4f);
        /// <summary>
        /// Controls how much the ambient occlusion affects direct lighting.
        /// </summary>
        public ClampedFloatParameter directLightingStrength = new ClampedFloatParameter(0f, 0f, 1f);
        /// <summary>
        /// Sampling radius. Bigger the radius, wider AO will be achieved, risking to lose fine details and increasing cost of the effect due to increasing cache misses.
        /// </summary>
        public ClampedFloatParameter radius = new ClampedFloatParameter(2.0f, 0.25f, 5.0f);
        /// <summary>
        /// Whether the results are accumulated over time or not. This can get higher quality results at a cheaper cost, but it can lead to temporal artifacts such as ghosting.
        /// </summary>
        public BoolParameter temporalAccumulation = new BoolParameter(true);

        // Temporal only parameters
        /// <summary>
        /// Moving this factor closer to 0 will increase the amount of accepted samples during temporal accumulation, increasing the ghosting, but reducing the temporal noise.
        /// </summary>
        public ClampedFloatParameter ghostingReduction = new ClampedFloatParameter(0.5f, 0.0f, 1.0f);

        // Non-temporal only parameters
        /// <summary>
        /// Modify the non-temporal blur to change how sharp features are preserved. Lower values leads to blurrier/softer results, higher values gets a sharper result, but with the risk of noise.
        /// </summary>
        public ClampedFloatParameter blurSharpness = new ClampedFloatParameter(0.1f, 0.0f, 1.0f);

        // Ray tracing parameters
        /// <summary>
        /// Defines the layers that ray traced ambient occlusion should include.
        /// </summary>
        public LayerMaskParameter layerMask = new LayerMaskParameter(-1);

        /// <summary>
        /// Controls the length of ray traced ambient occlusion rays.
        /// </summary>
        public ClampedFloatParameter rayLength = new ClampedFloatParameter(0.5f, 0f, 50f);

        /// <summary>
        /// Number of samples for evaluating the effect.
        /// </summary>
        public ClampedIntParameter sampleCount = new ClampedIntParameter(4, 1, 64);

        /// <summary>
        /// Defines if the ray traced ambient occlusion should be denoised.
        /// </summary>
        public BoolParameter denoise = new BoolParameter(false);

        /// <summary>
        /// Controls the radius of the ray traced ambient occlusion denoiser.
        /// </summary>
        public ClampedFloatParameter denoiserRadius = new ClampedFloatParameter(0.5f, 0.001f, 1.0f);

        /// <summary>
        /// Number of steps to take along one signed direction during horizon search (this is the number of steps in positive and negative direction). Increasing the value can lead to detection
        /// of finer details, but is not a guarantee of higher quality otherwise. Also note that increasing this value will lead to higher cost.
        /// </summary>
        public int stepCount
        {
            get
            {
                if (!UsesQualitySettings())
                    return m_StepCount.value;
                else
                    return GetLightingQualitySettings().AOStepCount[(int)quality.value];
            }
            set { m_StepCount.value = value; }
        }

        /// <summary>
        /// If this option is set to true, the effect runs at full resolution. This will increases quality, but also decreases performance significantly.
        /// </summary>
        public bool fullResolution
        {
            get
            {
                if (!UsesQualitySettings())
                    return m_FullResolution.value;
                else
                    return GetLightingQualitySettings().AOFullRes[(int)quality.value];
            }
            set { m_FullResolution.value = value; }
        }

        /// <summary>
        /// This field imposes a maximum radius in pixels that will be considered. It is very important to keep this as tight as possible to preserve good performance.
        /// Note that the pixel value specified for this field is the value used for 1080p when *not* running the effect at full resolution, it will be scaled accordingly
        /// for other resolutions.
        /// </summary>
        public int maximumRadiusInPixels
        {
            get
            {
                if (!UsesQualitySettings())
                    return m_MaximumRadiusInPixels.value;
                else
                    return GetLightingQualitySettings().AOMaximumRadiusPixels[(int)quality.value];
            }
            set { m_MaximumRadiusInPixels.value = value; }
        }

        /// <summary>
        /// This upsample method preserves sharp edges better, however may result in visible aliasing and it is slightly more expensive.
        /// </summary>
        public bool bilateralUpsample
        {
            get
            {
                if (!UsesQualitySettings())
                    return m_BilateralUpsample.value;
                else
                    return GetLightingQualitySettings().AOBilateralUpsample[(int)quality.value];
            }
            set { m_BilateralUpsample.value = value; }
        }

        /// <summary>
        /// Number of directions searched for occlusion at each each pixel when temporal accumulation is disabled.
        /// </summary>
        public int directionCount
        {
            get
            {
                if (!UsesQualitySettings())
                    return m_DirectionCount.value;
                else
                    return GetLightingQualitySettings().AODirectionCount[(int)quality.value];
            }
            set { m_DirectionCount.value = value; }
        }

        [SerializeField, FormerlySerializedAs("stepCount")]
        private ClampedIntParameter m_StepCount = new ClampedIntParameter(6, 2, 32);

        [SerializeField, FormerlySerializedAs("fullResolution")]
        private BoolParameter m_FullResolution = new BoolParameter(false);

        [SerializeField, FormerlySerializedAs("maximumRadiusInPixels")]
        private ClampedIntParameter m_MaximumRadiusInPixels = new ClampedIntParameter(40, 16, 256);

        // Temporal only parameter
        [SerializeField, FormerlySerializedAs("bilateralUpsample")]
        private BoolParameter m_BilateralUpsample = new BoolParameter(true);

        // Non-temporal only parameters
        [SerializeField, FormerlySerializedAs("directionCount")]
        private ClampedIntParameter m_DirectionCount = new ClampedIntParameter(2, 1, 6);
    }

    partial class AmbientOcclusionSystem
    {
        RenderPipelineResources m_Resources;
        RenderPipelineSettings m_Settings;

        private bool m_HistoryReady = false;
        private RTHandle m_PackedDataTex;
        private RTHandle m_PackedDataBlurred;
        private RTHandle m_AmbientOcclusionTex;
        private RTHandle m_FinalHalfRes;
        private RTHandle m_RawHarmonics1Tex;
        private RTHandle m_RawHarmonics5Tex;
        private RTHandle m_AmbientOcclusionSH1Tex;
        private RTHandle m_AmbientOcclusionSH5Tex;

        private bool m_RunningFullRes = false;
        private bool m_RunningSSLVD = false;

        readonly HDRaytracingAmbientOcclusion m_RaytracingAmbientOcclusion = new HDRaytracingAmbientOcclusion();

        private void ReleaseRT()
        {
            RTHandles.Release(m_AmbientOcclusionTex);
            RTHandles.Release(m_PackedDataTex);
            RTHandles.Release(m_PackedDataBlurred);
            RTHandles.Release(m_FinalHalfRes);
            RTHandles.Release(m_RawHarmonics1Tex);
            RTHandles.Release(m_RawHarmonics5Tex);
            RTHandles.Release(m_AmbientOcclusionSH1Tex);
            RTHandles.Release(m_AmbientOcclusionSH5Tex);
        }

        void AllocRT(float scaleFactor, bool enableLocalVisibilityDistribution)
        {
            m_AmbientOcclusionTex = RTHandles.Alloc(Vector2.one, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: GraphicsFormat.R8_UNorm, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "Ambient Occlusion");
            if (!enableLocalVisibilityDistribution)
            {
                m_PackedDataTex = RTHandles.Alloc(Vector2.one * scaleFactor, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: GraphicsFormat.R32_SFloat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Packed data");
                m_PackedDataBlurred = RTHandles.Alloc(Vector2.one * scaleFactor, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: GraphicsFormat.R32_SFloat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Packed blurred data");
                m_FinalHalfRes = RTHandles.Alloc(Vector2.one * 0.5f, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: GraphicsFormat.R32_SFloat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "Final Half Res AO Packed");
            }
            else
            {
                m_PackedDataTex = RTHandles.Alloc(Vector2.one * scaleFactor, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: GraphicsFormat.R32_SFloat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Packed data");
                m_RawHarmonics1Tex = RTHandles.Alloc(Vector2.one * scaleFactor, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: AmbientOcclusion.HarmonicsFormat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Harmonics Coeffs 1-4 raw");
                m_RawHarmonics5Tex = RTHandles.Alloc(Vector2.one * scaleFactor, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: AmbientOcclusion.HarmonicsFormat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Harmonics Coeffs 5-8 raw");
                m_AmbientOcclusionSH1Tex = RTHandles.Alloc(Vector2.one, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: AmbientOcclusion.HarmonicsFormat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Harmonics Coeffs 1-4");
                m_AmbientOcclusionSH5Tex = RTHandles.Alloc(Vector2.one, TextureXR.slices, filterMode: FilterMode.Point, colorFormat: AmbientOcclusion.HarmonicsFormat, dimension: TextureXR.dimension, useDynamicScale: true, enableRandomWrite: true, name: "AO Harmonics Coeffs 5-8");
            }
        }

        void EnsureRTSize(AmbientOcclusion settings, HDCamera hdCamera)
        {
            float scaleFactor = m_RunningFullRes ? 1.0f : 0.5f;
            if (settings.fullResolution != m_RunningFullRes || settings.localVisibilityDistribution.value != m_RunningSSLVD)
            {
                ReleaseRT();

                m_RunningFullRes = settings.fullResolution;
                m_RunningSSLVD = settings.localVisibilityDistribution.value;
                scaleFactor = m_RunningFullRes ? 1.0f : 0.5f;
                AllocRT(scaleFactor, settings.localVisibilityDistribution.value);
            }

            hdCamera.AllocateAmbientOcclusionHistoryBuffer(scaleFactor);
        }

        internal AmbientOcclusionSystem(HDRenderPipelineAsset hdAsset, RenderPipelineResources defaultResources)
        {
            m_Settings = hdAsset.currentPlatformRenderPipelineSettings;
            m_Resources = defaultResources;

            if (!hdAsset.currentPlatformRenderPipelineSettings.supportSSAO)
                return;

            AllocRT(0.5f, false);
        }

        internal void Cleanup()
        {
            if (HDRenderPipeline.GatherRayTracingSupport(m_Settings))
            {
                m_RaytracingAmbientOcclusion.Release();
            }

            ReleaseRT();
        }

        internal void InitRaytracing(HDRenderPipeline renderPipeline)
        {
            m_RaytracingAmbientOcclusion.Init(renderPipeline);
        }

        internal bool IsActive(HDCamera camera, AmbientOcclusion settings) => camera.frameSettings.IsEnabled(FrameSettingsField.SSAO) && settings.intensity.value > 0f;

        internal void Render(CommandBuffer cmd, HDCamera camera, ScriptableRenderContext renderContext, int frameCount)
        {
            var settings = camera.volumeStack.GetComponent<AmbientOcclusion>();

            if (!IsActive(camera, settings))
            {
                PostDispatchWork(cmd, camera);
                return;
            }
            else
            {
                if (camera.frameSettings.IsEnabled(FrameSettingsField.RayTracing) && settings.rayTracing.value)
                    m_RaytracingAmbientOcclusion.RenderAO(camera, cmd, m_AmbientOcclusionTex, renderContext, frameCount);
                else
                {
                    Dispatch(cmd, camera, frameCount);
                    PostDispatchWork(cmd, camera);
                }
            }
        }

        struct RenderAOParameters
        {
            public ComputeShader    aoCS;
            public int              aoKernel;
            public ComputeShader    denoiseAOCS;
            public int              denoiseKernelSpatial;

            public int              denoiseKernelTemporal;
            public int              denoiseKernelCopyHistory;
            public ComputeShader    upsampleAndBlurAOCS;
            public int              upsampleAndBlurKernel;
            public int              upsampleAOKernel;

            public Vector4          aoParams0;
            public Vector4          aoParams1;
            public Vector4          aoParams2;
            public Vector4          aoParams3;
            public Vector4          aoParams4;
            public Vector4          firstAndSecondMipOffsets;
            public Vector4          aoBufferInfo;
            public Vector4          toViewSpaceProj;
            public Vector2          runningRes;
            public Vector2          actualRes;
            public int              viewCount;
            public bool             historyReady;
            public int              outputWidth;
            public int              outputHeight;
            public bool             fullResolution;
            public bool             runAsync;
            public bool             motionVectorDisabled;
            public bool             temporalAccumulation;
            public bool             bilateralUpsample;
            public bool             localVisibilityDistribution;
        }

        RenderAOParameters PrepareRenderAOParameters(HDCamera camera, RTHandleProperties rtHandleProperties, Vector2 historySize, int frameCount)
        {
            var parameters = new RenderAOParameters();

            // Grab current settings
            var settings = camera.volumeStack.GetComponent<AmbientOcclusion>();
            parameters.fullResolution = settings.fullResolution;

            parameters.actualRes = new Vector2(camera.actualWidth, camera.actualHeight);
            if (parameters.fullResolution)
            {
                parameters.runningRes = parameters.actualRes;
                parameters.aoBufferInfo = new Vector4(camera.actualWidth, camera.actualHeight, 1.0f / camera.actualWidth, 1.0f / camera.actualHeight);
            }
            else
            {
                parameters.runningRes = parameters.actualRes * 0.5f;
                parameters.aoBufferInfo = new Vector4(camera.actualWidth * 0.5f, camera.actualHeight * 0.5f, 2.0f / camera.actualWidth, 2.0f / camera.actualHeight);
            }

            Vector2 usingRes = settings.localVisibilityDistribution.value ? parameters.actualRes : parameters.runningRes;

            float invHalfTanFOV = -camera.mainViewConstants.projMatrix[1, 1];
            float aspectRatio = usingRes.y / usingRes.x;

            parameters.aoParams0 = new Vector4(
                parameters.fullResolution ? 0.0f : 1.0f,
                usingRes.y * invHalfTanFOV * 0.25f,
                settings.radius.value,
                settings.stepCount
                );

            parameters.aoParams1 = new Vector4(
                settings.intensity.value,
                1.0f / (settings.radius.value * settings.radius.value),
                (frameCount / 6) % 4,
                settings.localVisibilityDistribution.value ? (frameCount % 16) : (frameCount % 6)
                );

            // We start from screen space position, so we bake in this factor the 1 / resolution as well.
            parameters.toViewSpaceProj = new Vector4(
                2.0f / (invHalfTanFOV * aspectRatio * usingRes.x),
                2.0f / (invHalfTanFOV * usingRes.y),
                1.0f / (invHalfTanFOV * aspectRatio),
                1.0f / invHalfTanFOV
                );

            float scaleFactor = (usingRes.x * usingRes.y) / (540.0f * 960.0f);
            float radInPixels = Mathf.Max(16, settings.maximumRadiusInPixels * Mathf.Sqrt(scaleFactor));

            parameters.aoParams2 = new Vector4(
                historySize.x,
                historySize.y,
                1.0f / (settings.stepCount + 1.0f),
                radInPixels
            );

            float stepSize = m_RunningFullRes ? 1 : 0.5f;

            float blurTolerance = 1.0f - settings.blurSharpness.value;
            float maxBlurTolerance = 0.25f;
            float minBlurTolerance = -2.5f;
            blurTolerance = minBlurTolerance + (blurTolerance * (maxBlurTolerance - minBlurTolerance));

            float bTolerance = 1f - Mathf.Pow(10f, blurTolerance) * stepSize;
            bTolerance *= bTolerance;
            const float upsampleTolerance = -7.0f; // TODO: Expose?
            float uTolerance = Mathf.Pow(10f, upsampleTolerance);
            float noiseFilterWeight = 1f / (Mathf.Pow(10f, 0.0f) + uTolerance);

            parameters.aoParams3 = new Vector4(
                bTolerance,
                uTolerance,
                noiseFilterWeight,
                stepSize
            );

            float upperNudgeFactor = 1.0f - settings.ghostingReduction.value;
            const float maxUpperNudgeLimit = 5.0f;
            const float minUpperNudgeLimit = 0.25f;
            upperNudgeFactor = minUpperNudgeLimit + (upperNudgeFactor * (maxUpperNudgeLimit - minUpperNudgeLimit));
            parameters.aoParams4 = new Vector4(
                settings.directionCount,
                upperNudgeFactor,
                minUpperNudgeLimit,
                invHalfTanFOV
            );

            var hdrp = (RenderPipelineManager.currentPipeline as HDRenderPipeline);
            var depthMipInfo = hdrp.sharedRTManager.GetDepthBufferMipChainInfo();
            parameters.firstAndSecondMipOffsets = new Vector4(depthMipInfo.mipLevelOffsets[1].x, depthMipInfo.mipLevelOffsets[1].y, depthMipInfo.mipLevelOffsets[2].x, depthMipInfo.mipLevelOffsets[2].y);

            parameters.bilateralUpsample = settings.bilateralUpsample;
            parameters.temporalAccumulation = settings.temporalAccumulation.value;
            parameters.localVisibilityDistribution = settings.localVisibilityDistribution.value;

            if (settings.localVisibilityDistribution.value)
            {
                parameters.aoCS = m_Resources.shaders.SSLVDCS;
                parameters.aoKernel = parameters.fullResolution ? parameters.aoCS.FindKernel("SSLVDMain_FullRes") : parameters.aoCS.FindKernel("SSLVDMain_HalfRes");

                parameters.denoiseAOCS = m_Resources.shaders.SSLVDDenoiseCS;
                parameters.denoiseKernelSpatial = parameters.fullResolution ? parameters.denoiseAOCS.FindKernel("SSLVDDenoise_Spatial") : parameters.denoiseAOCS.FindKernel("SSLVDDenoise_Spatial_HalfRes");
            }
            else
            {
                parameters.aoCS = m_Resources.shaders.GTAOCS;

                if (parameters.temporalAccumulation)
                {
                    if (parameters.fullResolution)
                    {
                        parameters.aoKernel = parameters.aoCS.FindKernel("GTAOMain_FullRes_Temporal");
                    }
                    else
                    {
                        parameters.aoKernel = parameters.aoCS.FindKernel("GTAOMain_HalfRes_Temporal");
                    }
                }
                else
                {
                    if (parameters.fullResolution)
                    {
                        parameters.aoKernel = parameters.aoCS.FindKernel("GTAOMain_FullRes");
                    }
                    else
                    {
                        parameters.aoKernel = parameters.aoCS.FindKernel("GTAOMain_HalfRes");
                    }
                }

                parameters.upsampleAndBlurAOCS = m_Resources.shaders.GTAOBlurAndUpsample;

                parameters.denoiseAOCS = m_Resources.shaders.GTAODenoiseCS;
                parameters.denoiseKernelSpatial = parameters.denoiseAOCS.FindKernel(parameters.temporalAccumulation ? "GTAODenoise_Spatial_To_Temporal" : "GTAODenoise_Spatial");

                parameters.denoiseKernelTemporal = parameters.denoiseAOCS.FindKernel(parameters.fullResolution ? "GTAODenoise_Temporal_FullRes" : "GTAODenoise_Temporal");
                parameters.denoiseKernelCopyHistory = parameters.denoiseAOCS.FindKernel("GTAODenoise_CopyHistory");

                parameters.upsampleAndBlurKernel = parameters.upsampleAndBlurAOCS.FindKernel("BlurUpsample");
                parameters.upsampleAOKernel = parameters.upsampleAndBlurAOCS.FindKernel(settings.bilateralUpsample ? "BilateralUpsampling" : "BoxUpsampling");
            }

            parameters.outputWidth = camera.actualWidth;
            parameters.outputHeight = camera.actualHeight;

            parameters.viewCount = camera.viewCount;
            parameters.historyReady = m_HistoryReady;
            m_HistoryReady = true; // assumes that if this is called, then render is done as well.

            parameters.runAsync = camera.frameSettings.SSAORunsAsync();
            parameters.motionVectorDisabled = !camera.frameSettings.IsEnabled(FrameSettingsField.MotionVectors);

            return parameters;
        }

        static void RenderAO(in RenderAOParameters  parameters,
                                RTHandle            aoTexture,
                                RTHandle            harmonics1Texture,
                                RTHandle            harmonics5Texture,
                                RenderPipelineResources resources,
                                CommandBuffer       cmd)
        {
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._AOBufferSize, parameters.aoBufferInfo);
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._AODepthToViewParams, parameters.toViewSpaceProj);
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._AOParams0, parameters.aoParams0);
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._AOParams1, parameters.aoParams1);
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._AOParams2, parameters.aoParams2);
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._AOParams4, parameters.aoParams4);
            cmd.SetComputeVectorParam(parameters.aoCS, HDShaderIDs._FirstTwoDepthMipOffsets, parameters.firstAndSecondMipOffsets);

            if (parameters.localVisibilityDistribution)
            {
                cmd.SetComputeTextureParam(parameters.aoCS, parameters.aoKernel, HDShaderIDs._AOPackedData, aoTexture);
                cmd.SetComputeTextureParam(parameters.aoCS, parameters.aoKernel, HDShaderIDs._AORawHarmonics1, harmonics1Texture);
                cmd.SetComputeTextureParam(parameters.aoCS, parameters.aoKernel, HDShaderIDs._AORawHarmonics5, harmonics5Texture);
            }
            else
            {
                cmd.SetComputeTextureParam(parameters.aoCS, parameters.aoKernel, HDShaderIDs._AOPackedData, aoTexture);
            }

            BlueNoise blueNoise = (RenderPipelineManager.currentPipeline as HDRenderPipeline).GetBlueNoiseManager();
            cmd.SetGlobalTexture(HDShaderIDs._BlueNoiseTexture, blueNoise.textureArray16L);

            const int groupSizeX = 8;
            const int groupSizeY = 8;
            int threadGroupX = ((int)parameters.runningRes.x + (groupSizeX - 1)) / groupSizeX;
            int threadGroupY = ((int)parameters.runningRes.y + (groupSizeY - 1)) / groupSizeY;

            cmd.DispatchCompute(parameters.aoCS, parameters.aoKernel, threadGroupX, threadGroupY, parameters.viewCount);
        }

        static void DenoiseAO(  in RenderAOParameters   parameters,
                                RTHandle                packedDataTex,
                                RTHandle                rawHarmonics1Tex,
                                RTHandle                rawHarmonics5Tex,
                                RTHandle                packedDataBlurredTex,
                                RTHandle                packedHistoryTex,
                                RTHandle                packedHistoryOutputTex,
                                RTHandle                aoOutputTex,
                                RTHandle                harmonics1OutputTex,
                                RTHandle                harmonics5OutputTex,
                                CommandBuffer           cmd)
        {
            const int groupSizeX = 8;
            const int groupSizeY = 8;

            if (parameters.localVisibilityDistribution)
            {
                var blurCS = parameters.denoiseAOCS;
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AODepthToViewParams, parameters.toViewSpaceProj);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AOParams0, parameters.aoParams0);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AOParams1, parameters.aoParams1);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AOParams2, parameters.aoParams2);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AOParams3, parameters.aoParams3);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AOParams4, parameters.aoParams4);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._AOBufferSize, parameters.aoBufferInfo);
                cmd.SetComputeVectorParam(blurCS, HDShaderIDs._FirstTwoDepthMipOffsets, parameters.firstAndSecondMipOffsets);

                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._AOPackedData, packedDataTex);
                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._AORawHarmonics1, rawHarmonics1Tex);
                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._AORawHarmonics5, rawHarmonics5Tex);
                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._OcclusionTexture, aoOutputTex);
                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._OcclusionHarmonics1Texture, harmonics1OutputTex);
                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._OcclusionHarmonics5Texture, harmonics5OutputTex);

                int groupsX = ((int)parameters.actualRes.x + (groupSizeX - 1)) / groupSizeX;
                int groupsY = ((int)parameters.actualRes.y + (groupSizeY - 1)) / groupSizeY;
                cmd.DispatchCompute(blurCS, parameters.denoiseKernelSpatial, groupsX, groupsY, parameters.viewCount);

                // Note!
                return;
            }

            int threadGroupX = ((int)parameters.runningRes.x + (groupSizeX - 1)) / groupSizeX;
            int threadGroupY = ((int)parameters.runningRes.y + (groupSizeY - 1)) / groupSizeY;

            if (parameters.temporalAccumulation || parameters.fullResolution)
            {
                var blurCS = parameters.denoiseAOCS;
                cmd.SetComputeVectorParam(parameters.denoiseAOCS, HDShaderIDs._AOParams1, parameters.aoParams1);
                cmd.SetComputeVectorParam(parameters.denoiseAOCS, HDShaderIDs._AOParams2, parameters.aoParams2);
                cmd.SetComputeVectorParam(parameters.denoiseAOCS, HDShaderIDs._AOParams3, parameters.aoParams3);
                cmd.SetComputeVectorParam(parameters.denoiseAOCS, HDShaderIDs._AOParams4, parameters.aoParams4);
                cmd.SetComputeVectorParam(parameters.denoiseAOCS, HDShaderIDs._AOBufferSize, parameters.aoBufferInfo);

                // Spatial
                cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._AOPackedData, packedDataTex);
                if (parameters.temporalAccumulation)
                {
                    cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._AOPackedBlurred, packedDataBlurredTex);
                }
                else
                {
                    cmd.SetComputeTextureParam(blurCS, parameters.denoiseKernelSpatial, HDShaderIDs._OcclusionTexture, aoOutputTex);
                }

                cmd.DispatchCompute(blurCS, parameters.denoiseKernelSpatial, threadGroupX, threadGroupY, parameters.viewCount);
            }

            if (parameters.temporalAccumulation)
            {
                if (!parameters.historyReady)
                {
                    cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelCopyHistory, HDShaderIDs._InputTexture, packedDataTex);      // TODO should use blurred as input here?
                    cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelCopyHistory, HDShaderIDs._OutputTexture, packedHistoryTex);
                    cmd.DispatchCompute(parameters.denoiseAOCS, parameters.denoiseKernelCopyHistory, threadGroupX, threadGroupY, parameters.viewCount);
                }

                // Temporal
                cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelTemporal, HDShaderIDs._AOPackedData, packedDataTex);
                cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelTemporal, HDShaderIDs._AOPackedBlurred, packedDataBlurredTex);
                cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelTemporal, HDShaderIDs._AOPackedHistory, packedHistoryTex);
                cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelTemporal, HDShaderIDs._AOOutputHistory, packedHistoryOutputTex);
                cmd.SetComputeTextureParam(parameters.denoiseAOCS, parameters.denoiseKernelTemporal, HDShaderIDs._OcclusionTexture, aoOutputTex);
                cmd.DispatchCompute(parameters.denoiseAOCS, parameters.denoiseKernelTemporal, threadGroupX, threadGroupY, parameters.viewCount);
            }
        }

        static void UpsampleAO( in RenderAOParameters   parameters,
                                RTHandle                input,
                                RTHandle                output,
                                CommandBuffer           cmd)
        {
            bool blurAndUpsample = !parameters.temporalAccumulation;

            cmd.SetComputeVectorParam(parameters.upsampleAndBlurAOCS, HDShaderIDs._AOBufferSize, parameters.aoBufferInfo);
            cmd.SetComputeVectorParam(parameters.upsampleAndBlurAOCS, HDShaderIDs._AOParams1, parameters.aoParams1);
            cmd.SetComputeVectorParam(parameters.upsampleAndBlurAOCS, HDShaderIDs._AOParams3, parameters.aoParams3);

            if (blurAndUpsample)
            {
                cmd.SetComputeTextureParam(parameters.upsampleAndBlurAOCS, parameters.upsampleAndBlurKernel, HDShaderIDs._AOPackedData, input);
                cmd.SetComputeTextureParam(parameters.upsampleAndBlurAOCS, parameters.upsampleAndBlurKernel, HDShaderIDs._OcclusionTexture, output);

                const int groupSizeX = 8;
                const int groupSizeY = 8;
                int threadGroupX = ((int)(parameters.runningRes.x) + (groupSizeX - 1)) / groupSizeX;
                int threadGroupY = ((int)(parameters.runningRes.y) + (groupSizeY - 1)) / groupSizeY;
                cmd.DispatchCompute(parameters.upsampleAndBlurAOCS, parameters.upsampleAndBlurKernel, threadGroupX, threadGroupY, parameters.viewCount);

            }
            else
            {
                cmd.SetComputeTextureParam(parameters.upsampleAndBlurAOCS, parameters.upsampleAOKernel, HDShaderIDs._AOPackedData, input);
                cmd.SetComputeTextureParam(parameters.upsampleAndBlurAOCS, parameters.upsampleAOKernel, HDShaderIDs._OcclusionTexture, output);

                const int groupSizeX = 8;
                const int groupSizeY = 8;
                int threadGroupX = ((int)parameters.runningRes.x + (groupSizeX - 1)) / groupSizeX;
                int threadGroupY = ((int)parameters.runningRes.y + (groupSizeY - 1)) / groupSizeY;
                cmd.DispatchCompute(parameters.upsampleAndBlurAOCS, parameters.upsampleAOKernel, threadGroupX, threadGroupY, parameters.viewCount);
            }
        }

        internal void Dispatch(CommandBuffer cmd, HDCamera camera, int frameCount)
        {
            var settings = camera.volumeStack.GetComponent<AmbientOcclusion>();
            if (IsActive(camera, settings))
            {
                using (new ProfilingScope(cmd, ProfilingSampler.Get(HDProfileId.RenderSSAO)))
                {
                    EnsureRTSize(settings, camera);

                    var currentHistory = camera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion);
                    var historyOutput = camera.GetPreviousFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion);

                    Vector2 historySize = new Vector2(currentHistory.referenceSize.x * currentHistory.scaleFactor.x,
                                                      currentHistory.referenceSize.y * currentHistory.scaleFactor.y);
                    var rtScaleForHistory = camera.historyRTHandleProperties.rtHandleScale;

                    var aoParameters = PrepareRenderAOParameters(camera, RTHandles.rtHandleProperties, historySize * rtScaleForHistory, frameCount);
                    using (new ProfilingScope(cmd, ProfilingSampler.Get(HDProfileId.HorizonSSAO)))
                    {
                        RenderAO(aoParameters, m_PackedDataTex, m_RawHarmonics1Tex, m_RawHarmonics5Tex, m_Resources, cmd);
                    }

                    using (new ProfilingScope(cmd, ProfilingSampler.Get(HDProfileId.DenoiseSSAO)))
                    {
                        var output = (m_RunningFullRes || m_RunningSSLVD) ? m_AmbientOcclusionTex : m_FinalHalfRes;
                        var outputSH1 = m_AmbientOcclusionSH1Tex;
                        var outputSH5 = m_AmbientOcclusionSH5Tex;
                        DenoiseAO(aoParameters, m_PackedDataTex, m_RawHarmonics1Tex, m_RawHarmonics5Tex, m_PackedDataBlurred, currentHistory, historyOutput, output, outputSH1, outputSH5, cmd);
                    }

                    if (!settings.localVisibilityDistribution.value && !m_RunningFullRes)
                    {
                        using (new ProfilingScope(cmd, ProfilingSampler.Get(HDProfileId.UpSampleSSAO)))
                        {
                            UpsampleAO(aoParameters, settings.temporalAccumulation.value ? m_FinalHalfRes : m_PackedDataTex, m_AmbientOcclusionTex, cmd);
                        }
                    }
                }
            }
        }

        internal void PushGlobalParameters(HDCamera hdCamera, CommandBuffer cmd)
        {
            var settings = hdCamera.volumeStack.GetComponent<AmbientOcclusion>();
            if (IsActive(hdCamera, settings))
                cmd.SetGlobalVector(HDShaderIDs._AmbientOcclusionParam, new Vector4(0f, 0f, 0f, settings.directLightingStrength.value));
            else
                cmd.SetGlobalVector(HDShaderIDs._AmbientOcclusionParam, Vector4.zero);
        }

        internal void PostDispatchWork(CommandBuffer cmd, HDCamera camera)
        {
            var settings = camera.volumeStack.GetComponent<AmbientOcclusion>();
            var aoTexture = TextureXR.GetBlackTexture();
            var aoSh1Texture = TextureXR.GetBlackTexture();
            var aoSh5Texture = TextureXR.GetBlackTexture();
            if (IsActive(camera, settings))
            {
                aoTexture = m_AmbientOcclusionTex;
                aoSh1Texture = settings.localVisibilityDistribution.value ? m_AmbientOcclusionSH1Tex : aoSh1Texture;
                aoSh5Texture = settings.localVisibilityDistribution.value ? m_AmbientOcclusionSH5Tex : aoSh5Texture;
            }
            cmd.SetGlobalTexture(HDShaderIDs._AmbientOcclusionTexture, aoTexture);
            cmd.SetGlobalTexture(HDShaderIDs._AmbientOcclusionSH1Texture, aoSh1Texture);
            cmd.SetGlobalTexture(HDShaderIDs._AmbientOcclusionSH5Texture, aoSh5Texture);
            // TODO: All the push debug stuff should be centralized somewhere
            (RenderPipelineManager.currentPipeline as HDRenderPipeline).PushFullScreenDebugTexture(camera, cmd, aoTexture, FullScreenDebugMode.SSAO);
        }
    }
}
