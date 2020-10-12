package cn.sxd.linearclassification

import cn.sxd.admm.SparseADMMOptimizer
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD


class SparseSVMWithADMM(numIterations: Int,
                        lambda: Double,
                        rho: Double,
                        cee: Double,
                        updateRho: Boolean,
                        used_Resid_primal_Update: Boolean,
                        used_L1_Update: Boolean
                       )
  extends GeneralizedLinearAlgorithm[SVMModel]
    with Serializable {

  override val optimizer = new SparseADMMOptimizer(
    numIterations,
    updateRho,
    used_Resid_primal_Update,
    used_L1_Update,
    SparseSVMWithADMMUpdater(lambda,rho, cee))

  override val validators = List(DataValidators.binaryLabelValidator)

  override def createModel(
                            weights: linalg.Vector,
                            intercept: Double): SVMModel = new SVMModel(weights, intercept)
}

object SparseSVMWithADMM {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             lambda: Double,
             rho: Double,
             cee: Double,
             updateRho: Boolean,
             used_Resid_primal_Update: Boolean,
             used_L1_Update: Boolean
           ): SVMModel = {
    new SparseSVMWithADMM(numIterations,lambda,rho, cee, updateRho, used_Resid_primal_Update, used_L1_Update).run(input)
  }

  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             lambda: Double,
             rho: Double,
             cee: Double
           ): SVMModel = {
    train(input, numIterations, lambda,rho, cee, false, true, false)
  }

}