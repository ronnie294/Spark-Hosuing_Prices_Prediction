package com.mr.spark.housingprice


import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


object housingPrediction {


  def main(args: Array[String]) {

    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    if (args.length != 2) {
      logger.error("Usage:\nwc.WordCountMain <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Housing Price Prediction")
    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()
    import sparkSession.implicits._


    var inputDataRDD = sc.textFile(args(0) + "/train.csv")
    var testDataRDD = sc.textFile(args(0) + "/test.csv")
    val header = inputDataRDD.first()

    inputDataRDD = inputDataRDD.filter(row => row != header)
    val testHeader = testDataRDD.first()
    testDataRDD = testDataRDD.filter(row => row != testHeader)
    val inputDataSplitRDD = inputDataRDD.map(line => line.split(","))
    val testDataSplitRDD = testDataRDD.map(line => line.split(","))

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

    val testDataMapRDD = testDataSplitRDD.map(p => Record(p(0).toDouble, p
    (1), parseDouble(p(2)), p(3).toDouble, p(4), p(5), p(6), p(7), p(8), p(9),
      p(10), p(11), p(12), p(13), p(14), p(15), p(16).toInt, p(17).toInt, p
      (18).toInt, p(19).toInt,
      p(20), p(21), p(22), p(23), p(24), p(25), p(26), p(27), p(28), p(29),
      p(30), p(31), p(32), parseDouble(p(33)), p(35), parseDouble(p(36)),
      parseDouble(p(37)),
      parseDouble(p(37)), p(38), p(39),
      p(40), p(41), p(42).toDouble, p(43).toDouble, p(44).toDouble, p(45)
        .toDouble, parseInt(p(46)), parseInt(p(47)), parseInt(p(48)),
      parseInt(p(49)),
      p(50).toInt, p(51).toInt, p(52), p(53).toInt, p(54), p(55).toInt, p(56)
      , p(57), parseInt(p(58)), p(59),
      parseInt(p(60)), parseDouble(p(61)), p(62), p(63), p(64), parseDouble(p
      (65)), p(66)
        .toDouble, p(67).toDouble, p(68).toDouble, p(69).toDouble,
      p(70).toDouble, p(71), p(72), p(73), p(74), p(75).toInt, p(76).toInt, p
      (77), p(78), p(79).toInt, 0.0))

    val ecommDF = inputDataMapRDD.toDF()
    val testecommDF = testDataMapRDD.toDF()
    //print("utark"+ecommDF.head(10))

    val ecommDF1 = ecommDF.select(ecommDF("SalePrice").as("label"),
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

    val testecommDF1 = testecommDF.select(
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
      $"id")

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

    //assembler: org.apache.spark.ml.feature.VectorAssembler =
    // vecAssembler_4c5ea5e20741
    val ecommDF2 = assembler.transform(ecommDF1).select($"label", $"features")
    val testecommDF2 = assembler.transform(testecommDF1).select($"features")

    val lr = new LinearRegression()

    val lrModel = lr.fit(ecommDF2)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${
      lrModel
        .intercept
    }")

    val trainingSummary = lrModel.summary

   // print("Columnsssss")
    lrModel.transform(testecommDF2).select("prediction").rdd.saveAsTextFile(args(1))

    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

    trainingSummary.residuals.show()

    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}