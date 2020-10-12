package cn.sxd.regression

import cn.sxd.admm.{DenseADMMOptimizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint, LassoModel}
import org.apache.spark.rdd.RDD

class RegressionWithADMM(numIterations: Int,
                         lambda: Double,
                         rho: Double,
                         updateRho: Boolean,
                         used_Resid_primal_Update: Boolean,
                         used_L1_Update: Boolean
                      )
  extends GeneralizedLinearAlgorithm[LassoModel] with Serializable {

  //设置lambda和rho
  override def optimizer: Optimizer = new DenseADMMOptimizer(
    numIterations,
    updateRho,
    used_Resid_primal_Update,
    used_L1_Update,
    new RegressionWithADMMUpdater(lambda, rho))

//  override val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LassoModel(weights, intercept)
}
}
object RegressionWithADMM {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             lambda: Double, //L2正则项
             rho: Double,
             updateRho: Boolean,
             used_Resid_primal_Update: Boolean,
             used_L1_Update: Boolean
           ): LassoModel = { //ADMM p的参数
    new RegressionWithADMM(numIterations, lambda, rho, updateRho, used_Resid_primal_Update, used_L1_Update).run(input)
  }


  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             rho: Double,
             lambda: Double
           ): LassoModel = {
    //    new SparseSVMWithADMM(numIterations, rho, cee, updateRho, used_Resid_primal_Update).run(input)
    train(input, numIterations, rho, lambda, false, true, false)
  }
}

