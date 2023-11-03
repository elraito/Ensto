using Microsoft.ML;
using Microsoft.ML.Data;
using Ensto.DataStructures;
using Ensto.YoloParser;

namespace Ensto;

class OnnxModelScorer
{
    private readonly string imagesFolder;
    private readonly string modelLocation;
    private readonly MLContext mlContext;

    private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();

    public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.mlContext = mlContext;
    }

    public struct ImageNetSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }

    public struct TinyYoloModelSettings
    {
        public const string ModelInput = "data";

        public const string ModelOutput = "model_outputs0";
    }

    private ITransformer LoadModel(string modelLocation)
    {
        var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

        var pipeline = mlContext.Transforms.LoadImages(
            outputColumnName: "data",
            imageFolder: "",
            inputColumnName: nameof(ImageNetData.ImagePath))
                        .Append(mlContext.Transforms.ResizeImages(
                            outputColumnName: "data",
                            imageWidth: ImageNetSettings.imageWidth,
                            imageHeight: ImageNetSettings.imageHeight,
                            inputColumnName: "data"))
                        .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "data"))
                        .Append(mlContext.Transforms.ApplyOnnxModel(
                            modelFile: modelLocation,
                            outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput },
                            inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

        var model = pipeline.Fit(data);

        return model;
    }

    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        IDataView scoredData = model.Transform(testData);

        IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

        return probabilities;
    }

    public IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(modelLocation);

        return PredictDataUsingModel(data, model);
    }
}
