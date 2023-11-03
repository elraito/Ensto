namespace Ensto.DataStructures;

using Microsoft.ML.Data;

public class ImageNetPrediction
{
    [ColumnName("model_outputs0")]
    public float[] PredictedLabels;
}