using Ensto;
using Ensto.DataStructures;
using Ensto.YoloParser;
using Microsoft.ML;
using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing.Processing;
using Point = SixLabors.ImageSharp.Point;
using RectangleF = SixLabors.ImageSharp.RectangleF;
using Color = SixLabors.ImageSharp.Color;
using SixLabors.ImageSharp.Drawing;

var assetsRelativePath = @"../../../assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
// var modelFilePath = System.IO.Path.Combine(assetsPath, "Model", "tinyyolov2-8.onnx");
var modelFilePath = System.IO.Path.Combine(assetsPath, "Model", "model.onnx");
var imagesFolder = System.IO.Path.Combine(assetsPath, "images");
var outputFolder = System.IO.Path.Combine(assetsPath, "images", "output");

MLContext mlContext = new MLContext();

try
{
    IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
    images = images.Where(x => !x.ImagePath.Contains("DS_Store")); // macos finder file
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

    var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

    YoloOutputParser parser = new YoloOutputParser();

    var boundingBoxes =
        probabilities
        .Select(probability => parser.ParseOutputs(probability))
        .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

    for (var i = 0; i < images.Count(); i++)
    {
        string imageFileName = images.ElementAt(i).Label;
        IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

        LogDetectedObjects(imageFileName, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}


var builder = WebApplication.CreateBuilder(args);

// builder.Services.AddSingleton<MLContext>();

var app = builder.Build();

app.MapGet("/", () => "Ok!");

// app.Run();

string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = System.IO.Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}

void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
{
    using var image = Image.Load(System.IO.Path.Combine(inputImageLocation, imageName));

    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    foreach (var box in filteredBoundingBoxes)
    {
        var x = (uint)Math.Max(box.Dimensions.X, 0);
        var y = (uint)Math.Max(box.Dimensions.Y, 0);
        var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

        x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
        y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
        width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
        height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

        string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

        Font font = SystemFonts.CreateFont("Arial", 12, FontStyle.Bold);
        var size = TextMeasurer.MeasureSize(text, new TextOptions(font));
        Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

        var pen = Pens.Solid(box.BoxColor, 3.2f);
        var brush = new SolidBrush(box.BoxColor);

        image.Mutate(ctx =>
        {
            ctx.Fill(brush, new RectangleF(atPoint.X, atPoint.Y, size.Width, size.Height));
            ctx.DrawText(text, font, Color.Black, atPoint);
            ctx.Draw(pen, new RectangleF(x, y, width, height));
        });
    }

    if (!Directory.Exists(outputImageLocation))
    {
        Directory.CreateDirectory(outputImageLocation);
    }

    image.Save(System.IO.Path.Combine(outputImageLocation, imageName));
}

void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
{
    Console.WriteLine($"Image {imageName} results");

    foreach (var box in boundingBoxes)
    {
        Console.WriteLine($"Tag: {box.Label} Confidence score: {box.Confidence}");
    }

    Console.WriteLine("");
}
