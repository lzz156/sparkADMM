package cn.sxd.admm

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, Vector, VectorBuilder}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
//trait关键字在scala中代表接口
trait ADMMUpdater {
  var rho:Double
  def L1Update(states: RDD[(ADMMState)]):SparseVector[Double]
  def L2Update(states: RDD[(ADMMState)]):DenseVector[Double]
  def uUpdate(state: ADMMState,z:Vector[Double]): ADMMState = {
    //逻辑回归的 y的更新
//   println("rho:"+rho)
    state.copy( y = state.y + rho*(state.x - z))
  }





}

//定义了停止准则
object ADMMUpdater {

  //全局变量Z的更新,dim:Int
  def linearZUpdate(lambda:Double,rho:Double)(states: RDD[(ADMMState)])={
    val numStates = states.partitions.length
//    println("ADMMState的分区数"+numStates)
//    val WandYBar = states.map(state=>state._2.x+state._2.y/rho).treeReduce(_+_)/numStates.toDouble
    val WandYBar = states.map(state=>state.x+state.y/rho).treeReduce(_+_)
    val penalty = rho/(rho * numStates + 2 * lambda)
    val zNew = penalty*WandYBar
    zNew
  }


  // Soft threshold
   def soft_shresholdZUpdate(lambda:Double,rho:Double)(admmStates:RDD[(ADMMState)]) = {
    val numStates = admmStates.partitions.length
    //    println("ADMMState的分区数"+numStates)
    val WandUBar = (admmStates.map(state=>state.x+state.y/rho).treeReduce(_+_))/numStates.toDouble
    val penalty = lambda / (rho * numStates)
//     val z: DenseVector[Double] = DenseVector.zeros[Double](WandUBar.size)
    val builder = new VectorBuilder[Double](WandUBar.size)
    for (ind <- 0 until WandUBar.size) {
      val v = WandUBar(ind)
      if (v > penalty) {
//        z(ind) = v-penalty
        builder.add(ind, v - penalty)
      } else if (v < -penalty) {
        builder.add(ind, v + penalty)
//        z(ind) = v+penalty
      }
    }
     builder.toSparseVector(true, true)
//      SparseVector.apply(z.data)
  }

}


