package com.mr.spark.housingprice


import org.apache.log4j.LogManager
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Housing price prediction using MLlib. Only Linear regression using all
  * numeric features only.
  * ToDo: Feature extraction using correlation to predict.
  */
object housingPrediction {


  def main(args: Array[String]) {

    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    if (args.length != 2) {
      logger.error("Usage:\nwc.housingPrediction <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Housing Price Prediction")
    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .config("spark.memory.offHeap.enabled", true)
      .config("spark.memory.offHeap.size", "16g")
      .getOrCreate()
    import sparkSession.implicits._

    //load training data
    var inputDataRDD = sc.textFile(args(0) + "/train.csv")

    //Remove the header and split the rdd
    val header = inputDataRDD.first()
    val inputDataSplitRDD = inputDataRDD.filter(row => row != header).map(line => line.split(","))
    // val inputDataSplitRDD = inputDataRDD.map(line => line.split(","))

    def parseInt(a: String): Int = {
      if (a != "NA")
        return a.toInt
      return 0
    }

    def parseDouble(a: String): Double = {
      if (a != "NA")
        return a.toDouble
      return 0.0
    }

    //Map the rdd to Object type
    val inputDataMapRDD = inputDataSplitRDD.map(p => Record(parseDouble(p(0)), p
    (1), parseDouble(p(2)), parseDouble(p(3)), p(4), p(5), p(6), p(7), p(8),
      p(9),
      p(10), p(11), p(12), p(13), p(14), p(15), p(16).toInt, p(17).toInt, p
      (18).toInt, p(19).toInt,
      p(20), p(21), p(22), p(23), p(24), p(25), p(26), p(27), p(28), p(29),
      p(30), p(31), p(32), parseDouble(p(33)), p(34), parseDouble(p(35)),
      parseDouble(p(36)), parseDouble(p(37)), p(38), p(39),
      p(40), p(41), parseDouble(p(42)), parseDouble(p(43)), parseDouble(p(44)
      ), parseDouble(p(45)), parseInt(p(46)), p(47).toInt, p(48).toInt, p(49)
        .toInt,
      p(50).toInt, p(51).toInt, p(52), p(53).toInt, p(54), p(55).toInt, p(56)
      , p(57), parseInt(p(58)), p(59),
      p(60).toInt, p(61).toDouble, p(62), p(63), p(64), p(65).toDouble, p(66)
        .toDouble, p(67).toDouble, p(68).toDouble, p(69).toDouble,
      p(70).toDouble, p(71), p(72), p(73), p(74), p(75).toInt, p(76).toInt, p
      (77), p(78), p(79).toInt, p(80).toDouble))

    val inputDataFrame = inputDataMapRDD.toDF()
    //Name Columns
    val trainDF = inputDataFrame.select(inputDataFrame("SalePrice").as("label"),
      $"MSSubClass", $"LotFrontage", $"LotArea", $"OverallQual",
      $"OverallCond",
      $"YearBuilt",
      $"YearRemodAdd", $"BsmtFinSF1",

      $"BsmtFinSF2",
      $"BsmtUnfSF",
      $"TotalBsmtSF", $"FirstFlrSF",
      $"SecondFlrSF",
      $"LowQualFinSF",
      $"GrLivArea",
      $"BsmtFullBath",
      $"BsmtHalfBath",
      $"FullBath",
      $"HalfBath",
      $"BedroomAbvGr",
      $"KitchenAbvGr",
      $"TotRmsAbvGrd",
      $"Fireplaces",
      $"GarageYrBlt",
      $"GarageCars",
      $"GarageArea",
      $"WoodDeckSF",
      $"OpenPorchSF",
      $"EnclosedPorch",
      $"SsnPorch",
      $"ScreenPorch",
      $"PoolArea",
      $"MoSold",
      $"YrSold",
      $"id",
      $"SalePrice")

    //Take features to be used for training the model
    val assembler = new VectorAssembler().setInputCols(Array("MSSubClass",
      "LotFrontage",
      "LotArea",
      "OverallQual",
      "OverallCond",
      "YearBuilt",
      "YearRemodAdd",
      "BsmtFinSF1",
      "BsmtFinSF2",
      "BsmtUnfSF",
      "TotalBsmtSF",
      "FirstFlrSF",
      "SecondFlrSF",
      "LowQualFinSF",
      "GrLivArea",
      "BsmtFullBath",
      "BsmtHalfBath",
      "FullBath",
      "HalfBath",
      "BedroomAbvGr",
      "KitchenAbvGr",
      "TotRmsAbvGrd",
      "Fireplaces",
      "GarageYrBlt",
      "GarageCars",
      "GarageArea",
      "WoodDeckSF",
      "OpenPorchSF",
      "EnclosedPorch",
      "SsnPorch",
      "ScreenPorch",
      "PoolArea",
      "MoSold",
      "YrSold",
      "id"
    )).setOutputCol(
      "features")

    val trainDFTransformed = assembler.transform(trainDF).select($"label",
      $"features")


    //Variables needed for Random Forest
    val splitSeed = 5043
    val noOfTrees = 500
    val maxDepth = 30
    val Array(trainingData, testData) = trainDFTransformed.randomSplit(Array(0.7, 0.3), splitSeed)
    val classifier = new RandomForestRegressor().setImpurity("variance")
      .setMaxDepth(maxDepth).setNumTrees(noOfTrees).setFeatureSubsetStrategy("auto")
      .setSeed(splitSeed)

    val rfModel = classifier.fit(trainingData)
    val predictions = rfModel.transform(trainingData)
    predictions.select("prediction", "label", "features").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    rfModel.toDebugString

    //Linear Regression
    val lr = new LinearRegression()
    val lrModel = lr.fit(trainDFTransformed)
    val trainingSummary = lrModel.summary
    lrModel.transform(trainDFTransformed).select("prediction").rdd
      .saveAsTextFile(args(1))


    println(s"No of trees in Random Forest: ${noOfTrees}")
    println(s"Max depth for Random Forest: ${maxDepth}")
    println(s"Random Forest RMSE:  ${rmse}")
    println(s"Linear Regression RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"LinearMSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}