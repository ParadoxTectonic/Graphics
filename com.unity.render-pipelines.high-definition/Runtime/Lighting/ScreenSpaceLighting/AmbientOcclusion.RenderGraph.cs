using UnityEngine.Experimental.Rendering;
using UnityEngine.Experimental.Rendering.RenderGraphModule;

namespace UnityEngine.Rendering.HighDefinition
{
    partial class AmbientOcclusionSystem
    {
        public RenderGraphResource Render(RenderGraph renderGraph, HDCamera hdCamera, RenderGraphResource depthPyramid, RenderGraphResource motionVectors, int frameCount)
        {
            var settings = hdCamera.volumeStack.GetComponent<AmbientOcclusion>();

            RenderGraphResource result;
            // AO has side effects (as it uses an imported history buffer)
            // So we can't rely on automatic pass stripping. This is why we have to be explicit here.
            if (IsActive(hdCamera, settings))
            {
                {
                    EnsureRTSize(settings, hdCamera);

                    var aoParameters = PrepareRenderAOParameters(hdCamera, renderGraph.rtHandleProperties, frameCount);

                    var currentHistory = renderGraph.ImportTexture(hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion));
                    var outputHistory = renderGraph.ImportTexture(hdCamera.GetPreviousFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion));

                    RenderGraphResource packedData, rawHarmonics1, rawHarmonics5;
                    RenderAO(renderGraph, aoParameters, depthPyramid, out packedData, out rawHarmonics1, out rawHarmonics5);
                    result = DenoiseAO(renderGraph, aoParameters, motionVectors, packedData, rawHarmonics1, rawHarmonics5, currentHistory, outputHistory);
                }
            }
            else
            {
                result = renderGraph.ImportTexture(TextureXR.GetBlackTexture(), HDShaderIDs._AmbientOcclusionTexture);
            }
            return result;
        }

        class RenderAOPassData
        {
            public RenderAOParameters           parameters;
            public RenderGraphMutableResource   packedData;
            public RenderGraphMutableResource   rawHarmonics1;
            public RenderGraphMutableResource   rawHarmonics5;
            public RenderGraphResource          depthPyramid;
        }

        void RenderAO(RenderGraph renderGraph, in RenderAOParameters parameters, RenderGraphResource depthPyramid, out RenderGraphResource packedDataOut, out RenderGraphResource rawHarmonics1Out, out RenderGraphResource rawHarmonics5Out)
        {
            using (var builder = renderGraph.AddRenderPass<RenderAOPassData>("GTAO Horizon search and integration", out var passData, ProfilingSampler.Get(HDProfileId.HorizonSSAO)))
            {
                builder.EnableAsyncCompute(parameters.runAsync);

                float scaleFactor = parameters.fullResolution ? 1.0f : 0.5f;

                passData.parameters = parameters;
                if (parameters.localVisibilityDistribution)
                {
                    passData.packedData = builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one * scaleFactor, true, true)
                    { colorFormat = GraphicsFormat.R32_SFloat, enableRandomWrite = true, name = "AO Packed data" }));
                    passData.rawHarmonics1 = builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one * scaleFactor, true, true)
                    { colorFormat = AmbientOcclusion.HarmonicsFormat, enableRandomWrite = true, name = "AO Harmonics Coeffs 1-4" }));
                    passData.rawHarmonics5 = builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one * scaleFactor, true, true)
                    { colorFormat = AmbientOcclusion.HarmonicsFormat, enableRandomWrite = true, name = "AO Harmonics Coeffs 5-8" }));
                }
                else
                {
                    passData.packedData = builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one * scaleFactor, true, true)
                    { colorFormat = GraphicsFormat.R32_UInt, enableRandomWrite = true, name = "AO Packed data" }));
                }
                passData.depthPyramid = builder.ReadTexture(depthPyramid);

                builder.SetRenderFunc(
                (RenderAOPassData data, RenderGraphContext ctx) =>
                {
                    RenderAO(data.parameters, ctx.resources.GetTexture(data.packedData), ctx.resources.GetTexture(data.rawHarmonics1), ctx.resources.GetTexture(data.rawHarmonics5), m_Resources, ctx.cmd);
                });

                packedDataOut = passData.packedData;
                rawHarmonics1Out = passData.rawHarmonics1;
                rawHarmonics5Out = passData.rawHarmonics5;
            }
        }

        class DenoiseAOPassData
        {
            public RenderAOParameters           parameters;
            public RenderGraphResource          packedData;
            public RenderGraphResource          rawHarmonics1;
            public RenderGraphResource          rawHarmonics5;
            public RenderGraphMutableResource   packedDataBlurred;
            public RenderGraphResource          currentHistory;
            public RenderGraphMutableResource   outputHistory;
            public RenderGraphMutableResource   denoiseOutput;
            public RenderGraphMutableResource   denoiseOutputHarmonics1;
            public RenderGraphMutableResource   denoiseOutputHarmonics5;
            public RenderGraphResource          motionVectors;
        }

        RenderGraphResource DenoiseAO(  RenderGraph                 renderGraph,
                                        in RenderAOParameters       parameters,
                                        RenderGraphResource         motionVectors,
                                        RenderGraphResource         aoPackedData,
                                        RenderGraphResource         rawHarmonics1,
                                        RenderGraphResource         rawHarmonics5,
                                        RenderGraphMutableResource  currentHistory,
                                        RenderGraphMutableResource  outputHistory)
        {
            RenderGraphResource denoiseOutput;
            RenderGraphResource denoiseOutputHarmonics1;
            RenderGraphResource denoiseOutputHarmonics5;

            using (var builder = renderGraph.AddRenderPass<DenoiseAOPassData>("Denoise GTAO", out var passData))
            {
                builder.EnableAsyncCompute(parameters.runAsync);

                float scaleFactor = parameters.fullResolution ? 1.0f : 0.5f;

                passData.parameters = parameters;
                passData.packedData = builder.ReadTexture(aoPackedData);
                passData.motionVectors = builder.ReadTexture(motionVectors);
                if (!parameters.localVisibilityDistribution)
                {
                    passData.packedDataBlurred = builder.WriteTexture(renderGraph.CreateTexture(
                        new TextureDesc(Vector2.one * scaleFactor, true, true) { colorFormat = GraphicsFormat.R32_UInt, enableRandomWrite = true, name = "AO Packed blurred data" } ));
                    passData.currentHistory = builder.ReadTexture(currentHistory); // can also be written on first frame, but since it's an imported resource, it doesn't matter in term of lifetime.
                    passData.outputHistory = builder.WriteTexture(outputHistory);
                    passData.denoiseOutput = parameters.fullResolution ?
                        builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one, true, true)        { enableRandomWrite = true, colorFormat = GraphicsFormat.R8_UNorm, name = "Ambient Occlusion" }, HDShaderIDs._AmbientOcclusionTexture)) :
                        builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one * 0.5f, true, true) { enableRandomWrite = true, colorFormat = GraphicsFormat.R32_UInt, name = "Final Half Res AO Packed" }));
                }
                else
                {
                    passData.rawHarmonics1 = builder.ReadTexture(rawHarmonics1);
                    passData.rawHarmonics5 = builder.ReadTexture(rawHarmonics5);
                    passData.denoiseOutput =
                        builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one, true, true) { enableRandomWrite = true, colorFormat = GraphicsFormat.R8_UNorm, name = "Ambient Occlusion" }, HDShaderIDs._AmbientOcclusionTexture));
                    passData.denoiseOutputHarmonics1 = 
                        builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one, true, true) { enableRandomWrite = true, colorFormat = AmbientOcclusion.HarmonicsFormat, name = "Ambient Occlusion Harmonics 1-4" }, HDShaderIDs._AmbientOcclusionSH1Texture));
                    passData.denoiseOutputHarmonics5 =
                        builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one, true, true) { enableRandomWrite = true, colorFormat = AmbientOcclusion.HarmonicsFormat, name = "Ambient Occlusion Harmonics 5-8" }, HDShaderIDs._AmbientOcclusionSH5Texture));
                }

                denoiseOutput = passData.denoiseOutput;
                denoiseOutputHarmonics1 = passData.denoiseOutputHarmonics1;
                denoiseOutputHarmonics5 = passData.denoiseOutputHarmonics5;

                builder.SetRenderFunc(
                (DenoiseAOPassData data, RenderGraphContext ctx) =>
                {
                    var res = ctx.resources;
                    DenoiseAO(  data.parameters,
                                res.GetTexture(data.packedData),
                                res.GetTexture(data.rawHarmonics1),
                                res.GetTexture(data.rawHarmonics5),
                                res.GetTexture(data.packedDataBlurred),
                                res.GetTexture(data.currentHistory),
                                res.GetTexture(data.outputHistory),
                                res.GetTexture(data.denoiseOutput),
                                res.GetTexture(data.denoiseOutputHarmonics1),
                                res.GetTexture(data.denoiseOutputHarmonics5),
                                ctx.cmd);
                });

                if (parameters.fullResolution)
                    return passData.denoiseOutput;
            }

            return UpsampleAO(renderGraph, parameters, denoiseOutput, denoiseOutputHarmonics1, denoiseOutputHarmonics5);
        }

        class UpsampleAOPassData
        {
            public RenderAOParameters           parameters;
            public RenderGraphResource          input;
            public RenderGraphMutableResource   output;
        }

        RenderGraphResource UpsampleAO(RenderGraph renderGraph, in RenderAOParameters parameters, RenderGraphResource input, RenderGraphResource inputHarmonics1, RenderGraphResource inputHarmonics5)
        {
            using (var builder = renderGraph.AddRenderPass<UpsampleAOPassData>("Upsample GTAO", out var passData, ProfilingSampler.Get(HDProfileId.UpSampleSSAO)))
            {
                builder.EnableAsyncCompute(parameters.runAsync);

                passData.parameters = parameters;
                passData.input = builder.ReadTexture(input);
                passData.output = builder.WriteTexture(renderGraph.CreateTexture(new TextureDesc(Vector2.one, true, true) { enableRandomWrite = true, colorFormat = GraphicsFormat.R8_UNorm, name = "Ambient Occlusion" }, HDShaderIDs._AmbientOcclusionTexture));

                builder.SetRenderFunc(
                (UpsampleAOPassData data, RenderGraphContext ctx) =>
                {
                    UpsampleAO(data.parameters, ctx.resources.GetTexture(data.input), ctx.resources.GetTexture(data.output), ctx.cmd);
                });

                return passData.output;
            }
        }
    }
}
