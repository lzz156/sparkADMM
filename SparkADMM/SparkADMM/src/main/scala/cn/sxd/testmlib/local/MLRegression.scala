package cn.sxd.testmlib.local

import java.util.Date

import cn.sxd.util.MyBLAS.dot
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object MLRegression {

  def main(args: Array[String]): Unit = {
    val data_path = "D://data//YearPredictionMSD.dat"
    val test_data_path = "D://data//YearPredictionMSDt.dat"

    val spark = SparkSession.builder().appName("LinearRegression").master("local").getOrCreate()
    Logger.getRootLogger.setLevel(Level.WARN)


    val training = spark.read.format("libsvm").load(data_path)
    val start_time = new Date().getTime
    val lr = new LinearRegression()
    //最大迭代次数
    lr.setMaxIter(100)
    //正则化参数
    lr.setRegParam(1.0)
    //学习率
    lr.setElasticNetParam(1)
    val lrModel = lr.fit(training)
    val coefficients: linalg.Vector = lrModel.coefficients
    val rmodel: Vector = Vectors.dense(coefficients.toArray)
    val end_time = new Date().getTime
    val predictData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(spark.sparkContext, test_data_path, 90)
    val total: Int = predictData.collect().length
    val ybar: Double = predictData.map(lp => lp.label).reduce(_ + _) / total
    val SST: Double = predictData.map { lp =>

      val Vay = (lp.label - ybar)* (lp.label - ybar)
      Vay
    }.reduce(_ + _)

    val SSE = predictData.map { lp =>
      val prediction: Double = dot(rmodel, lp.features)
      val d = (lp.label-prediction)*(lp.label-prediction)
      d
    }reduce(_ +_ )

    val R: Double = 1-SSE/SST
    println("均方误差（Mean Squared Error, MSE）"+SSE/total)
    println("决定系数:"+R)
    println("run time:"+(end_time-start_time).toDouble/1000)
  }
}
