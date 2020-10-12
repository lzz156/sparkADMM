package cn.sxd.linearclassification

import cn.sxd.admm.{DenseADMMOptimizer, SparseADMMOptimizer}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

class DenseLRADMM(numIterations: Int,
                   lambda: Double,
                   rho: Double,
                   updateRho: Boolean,
                   used_Resid_primal_Update: Boolean,
                   used_L1_Update: Boolean
                                  )
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

    //设置lambda和rho
    override def optimizer: Optimizer = new DenseADMMOptimizer(
      numIterations,
      updateRho,
      used_Resid_primal_Update,
      used_L1_Update,
      DenseLRWithADMMUpdater(lambda, rho))

    override val validators = List(DataValidators.binaryLabelValidator)

    override protected def createModel(weights: linalg.Vector, intercept: Double): LogisticRegressionModel
    = new LogisticRegressionModel(weights, intercept)
  }

object DenseLRADMM {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             lambda: Double, //L2正则项
             rho: Double,
             updateRho: Boolean,
             used_Resid_primal_Update: Boolean,
             used_L1_Update:Boolean
           ): LogisticRegressionModel = { //ADMM p的参数
    new DenseLRADMM(numIterations, lambda, rho, updateRho, used_Resid_primal_Update,used_L1_Update).run(input)
  }


  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             rho: Double,
             lambda: Double
           ): LogisticRegressionModel = {
    //    new SparseSVMWithADMM(numIterations, rho, cee, updateRho, used_Resid_primal_Update).run(input)
    train(input, numIterations, rho, lambda, false, true,false)
  }
}
