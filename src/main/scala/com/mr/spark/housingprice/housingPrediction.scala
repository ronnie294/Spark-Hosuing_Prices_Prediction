package com.mr.spark.housingprice


import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col


object housingPrediction {

  case class Record(

                     MSSubclass: Double,
                     MSZoning: String,
                     LotFrontage: Double,
                     LotArea: Double,
                     Street: String,
                     Alley: String,
                     LotShape: String,
                     LandContour: String,
                     Utilities: String,
                     LotConfig: String,
                     LandSLope: String,
                     Neighbourhood: String,
                     Condition1: String,
                     Condition2: String,
                     BldgType: String,
                     HouseStyle: String,
                     OverallQual: Int,
                     OverallCond: Int,
                     YearBuilt: Int,
                     YearRemodAdd: Int,
                     RoofStyle: String,
                     RoofMatl: String,
                     Exterior1st: String,
                     Exterior2nd: String,
                     MasVnrType: String,
                     MasVnrArea: String,
                     ExterQual: String,
                     ExterCond: String,
                     Foundation: String,
                     BsmtQual: String,
                     BsmtCond: String,
                     BsmtExposure: String,
                     BsmtFinType1: String,
                     BsmtFinSF1: Double,
                     BsmtFinType2: String,
                     BsmtFinSF2: Double,
                     BsmtUnfSF: Double,
                     TotalBsmtSF: Double,
                     Heating: String,
                     HeatingQC: String,
                     CentralAir: String,
                     Electrical: String,
                     FirstFlrSF: Double,
                     SecondFlrSF: Double,
                     LowQualFinSF: Double,
                     GrLivArea: Double,
                     BsmtFullBath: Int,
                     BsmtHalfBath: Int,
                     FullBath: Int,
                     HalfBath: Int,
                     BedroomAbvGr: Int,
                     KitchenAbvGr: Int,
                     KitchenQual: String,
                     TotRmsAbvGrd: Int,
                     Functional: String,
                     Fireplaces: Int,
                     FireplaceQu: String,
                     GarageType: String,
                     GarageYrBlt: Int,
                     GarageFinish: String,
                     GarageCars: Int,
                     GarageArea: Double,
                     GarageQual: String,
                     GarageCond: String,
                     PavedDrive: String,
                     WoodDeckSF: Double,
                     OpenPorchSF: Double,
                     EnclosedPorch: Double,
                     SsnPorch: Double,
                     ScreenPorch: Double,
                     PoolArea: Double,
                     PoolQC: String,
                     Fence: String,
                     MiscFeature: String,
                     MiscVal: String,
                     MoSold: Int,
                     YrSold: Int,
                     SaleType: String,
                     SaleCondition: String,
                     id: Int,
                     SalePrice: Double
                   )

  def main(args: Array[String]) {

    val logger: org.apache.log4j.Logger = LogManager.getRootLogger

    if (args.length != 2) {
      logger.error("Usage:\nwc.WordCountMain <input dir> <output dir>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Word Count")
    val sc = new SparkContext(conf)
    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()
    import sparkSession.implicits._




    /* val textFile = sc.textFile(args(0))
     val counts = textFile.flatMap(line => line.split(" "))
       .map(word => (word, 1))
       .reduceByKey(_ + _)
     counts.saveAsTextFile(args(1))*/

    var inputDataRDD = sc.textFile(args(0) + "/train.csv")
    var testDataRDD = sc.textFile(args(0) + "/test.csv")
    val header = inputDataRDD.first()

    inputDataRDD = inputDataRDD.filter(row => row!=header)
    val testHeader = testDataRDD.first()
    testDataRDD = testDataRDD.filter(row => row!=testHeader)
    val inputDataSplitRDD = inputDataRDD.map(line => line.split(","))
    val testDataSplitRDD = testDataRDD.map(line => line.split(","))
    val inputDataMapRDD = inputDataSplitRDD.map(p => Record(p(0).toDouble, p
    (1), if(p(2)!="NA") p(2).toDouble else 0.0, p(3).toDouble, p(4), p(5), p(6), p(7), p(8), p(9),
      p(10), p(11), p(12), p(13), p(14), p(15), p(16).toInt, p(17).toInt, p
      (18).toInt, p(19).toInt,
      p(20), p(21), p(22), p(23), p(24), p(25), p(26), p(27), p(28), p(29),
      p(30), p(31), p(32), p(33).toDouble, p(34), p(35).toDouble, p(36)
        .toDouble, p(37).toDouble, p(38), p(39),
      p(40), p(41), p(42).toDouble, p(43).toDouble, p(44).toDouble, p(45)
        .toDouble, p(46).toInt, p(47).toInt, p(48).toInt, p(49).toInt,
      p(50).toInt, p(51).toInt, p(52), p(53).toInt, p(54), p(55).toInt, p(56)
      , p(57), if(p(58)!="NA") p(58).toInt else 0, p(59),
      p(60).toInt, p(61).toDouble, p(62), p(63), p(64), p(65).toDouble, p(66)
        .toDouble, p(67).toDouble, p(68).toDouble, p(69).toDouble,
      p(70).toDouble, p(71), p(72), p(73), p(74), p(75).toInt, p(76).toInt, p
      (77), p(78), p(79).toInt, p(80).toDouble))

    val testDataMapRDD = testDataSplitRDD.map(p => Record(p(0).toDouble, p
    (1), if(p(2)!="NA") p(2).toDouble else 0.0, p(3).toDouble, p(4), p(5), p(6), p(7), p(8), p(9),
      p(10), p(11), p(12), p(13), p(14), p(15), p(16).toInt, p(17).toInt, p
      (18).toInt, p(19).toInt,
      p(20), p(21), p(22), p(23), p(24), p(25), p(26), p(27), p(28), p(29),
      p(30), p(31), p(32), if(p(33)!="NA") p(33).toDouble else 0.0, p(35),if(p(35)!="NA") p(35).toDouble else 0.0,if(p(36)!="NA") p(36).toDouble else 0.0,
      if(p(37)!="NA") p(37).toDouble else 0.0, p(38), p(39),
      p(40), p(41), p(42).toDouble, p(43).toDouble, p(44).toDouble, p(45)
        .toDouble,  if(p(46)!="NA") p(46).toInt else 0,  if(p(47)!="NA") p(47).toInt else 0,  if(p(48)!="NA") p(48).toInt else 0,  if(p(49)!="NA") p(49).toInt else 0,
      p(50).toInt, p(51).toInt, p(52), p(53).toInt, p(54), p(55).toInt, p(56)
      , p(57), if(p(58)!="NA") p(58).toInt else 0, p(59),
      if(p(60)!="NA") p(60).toInt else 0,if(p(61)!="NA") p(61).toDouble else 0.0, p(62), p(63), p(64), if(p(65)!="NA") p(65).toDouble else 0.0, p(66)
        .toDouble, p(67).toDouble, p(68).toDouble, p(69).toDouble,
      p(70).toDouble, p(71), p(72), p(73), p(74), p(75).toInt, p(76).toInt, p
      (77), p(78), p(79).toInt,0.0))

    val ecommDF = inputDataMapRDD.toDF()
    val testecommDF = testDataMapRDD.toDF()
    print("utark"+ecommDF.head(10))

    val ecommDF1 = ecommDF.select(ecommDF("SalePrice").as("label"),
      $"MSSubClass",$"LotFrontage", $"LotArea", $"OverallQual",
      $"OverallCond",
      $"YearBuilt",
      $"YearRemodAdd",$"BsmtFinSF1",

      $"BsmtFinSF2",
      $"BsmtUnfSF",
      $"TotalBsmtSF",$"FirstFlrSF",
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
      $"MSSubClass",$"LotFrontage", $"LotArea", $"OverallQual",
      $"OverallCond",
      $"YearBuilt",
      $"YearRemodAdd",$"BsmtFinSF1",

      $"BsmtFinSF2",
      $"BsmtUnfSF",
      $"TotalBsmtSF",$"FirstFlrSF",
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
    val testecommDF2 = assembler.transform(testecommDF1).select( $"features")

    val lr = new LinearRegression()

    val lrModel = lr.fit(ecommDF2)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel
      .intercept}")

    val trainingSummary = lrModel.summary

    print("Columnsssss")
    lrModel.transform(testecommDF2).select("prediction").rdd.saveAsTextFile(args(1))

    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

    trainingSummary.residuals.show()

    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"MSE: ${trainingSummary.meanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
}