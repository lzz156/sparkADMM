package cn.sxd.admm

import breeze.linalg.{DenseMatrix, DenseVector, Vector, VectorBuilder, norm}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.rdd.RDD

import scala.util.control.Breaks


class DenseADMMOptimizer(numIterations: Int,
                         updateRho: Boolean,
                         used_Resid_primal_Update: Boolean,
                         used_L1_Update: Boolean,
                         updater: DenseMatrixUpdater
                   )
  extends Optimizer with Serializable {

  var resid_dual = 0.0
  var resid_primal = 0.0
  //  var (eps_primal,eps_dual):Tuple =0.0
  //计算原始容忍度
  var eps_primal = 0.0
  //计算对偶容忍度
  var eps_dual = 0.0

  override def optimize(data: RDD[(Double, linalg.Vector)], initialWeights: linalg.Vector): linalg.Vector = {
    val dim_x: Int = initialWeights.size

    //    println("只使用对偶残差更新:"+used_Resid_primal_Update)
    //    println("开启自动更新参数rho:"+updateRho)
    //判断是L1正则还是L2正则
    var admm_z = if (used_L1_Update) {
      new VectorBuilder[Double](dim_x).toSparseVector()
      //        SparseVector.zeros[Double](dim_x)
    } else {
      DenseVector.zeros[Double](dim_x)
    }

    val sc: SparkContext = data.context
    var broad_z = sc.broadcast(admm_z)

    println("dataSize:" + 50000)

    val partitionRDD: RDD[Array[(Double, linalg.Vector)]] = data.glom()
    partitionRDD.persist()

    //初始化特征矩阵
    val admmdata: RDD[(DenseMatrix[Double], DenseVector[Double])] = partitionRDD.map { data => {
      val x = data.map(x => x._2.toArray)
      val b = data.map(x => x._1)
      val y: DenseVector[Double] = new DenseVector[Double](b)
      val mat = DenseMatrix(x: _*)
      (mat, y)
    }
    }
    admmdata.persist()
    val numPartions = data.partitions.length
    println("numPartitions:" + numPartions)

    var admmStates = admmdata.map( _ => ADMMState(initialWeights.toArray))
    println("x lenght: " + dim_x)
    println("#iternum" + "\t" + "    resid_primal" + "\t" + "       eps_primal" + "\t" + "     resid_dual" + "\t" + "       eps_dual" + "\t" + "rho")

    val loop = new Breaks
    loop.breakable {
      for (i <- 0 to numIterations) {
        print(i + "\t")

        admmStates = admmdata.zip(admmStates).map {
          map => {
//            updater.xUpdate(map., map._2._2, broad_z.value)
            updater.xUpdate(map._1,map._2,broad_z.value)
            //            (map._1,updater.xUpdate(map._2._1.toArray,map._2._2,broad_z.value))
          }
        }


        admmStates.cache()


        //判断使用软阈值法跟新还是线性更新 全局变量z
        val new_z = if (used_L1_Update) {
          updater.L1Update(admmStates)
        } else {
          updater.L2Update(admmStates)
        }

        resid_dual = updater.rho * math.sqrt(numPartions) * norm(new_z - admm_z)
        admm_z = new_z
        broad_z = sc.broadcast(admm_z)
        //        admmStates.cache()

        //原始残差
        val resid = admmStates.map(state => state.x - broad_z.value)
        resid.cache()

        //更新原始残差
        resid_primal = math.sqrt(resid.map(x => math.pow(norm(x), 2)).sum())

        admmStates = admmStates.map {
          map =>
            ( updater.uUpdate(map, broad_z.value))
        }
        admmStates.cache()

        //计算原始误差和对偶误差
        val (eps_primal_i, eps_dual_i): (Double, Double) = compute_eps_primalAndeps_dual(admmStates, dim_x, numPartions, admm_z)
        eps_primal = eps_primal_i
        eps_dual = eps_dual_i

        println("\t" + resid_primal + "\t" + eps_primal + "\t" + resid_dual + "\t" + eps_dual + "\t" + updater.rho)

        // 只使用原始残差作为停止准则
        if (used_Resid_primal_Update) {
          if (resid_primal < eps_primal) {
            loop.break
          }
        } else {
          // 使用原始残差和对偶残差作为停止准则
          if (resid_primal < eps_primal && resid_dual < eps_dual) {
            loop.break
          }
        }

        //开启rho动态更新
        if (updateRho)
          update_rho
      }
    }
    linalg.Vectors.dense(admm_z.toDenseVector.data)
  }
  private def compute_eps_primalAndeps_dual(admmStates: RDD[(ADMMState)], dim_x: Int, npart: Int, admm_z: Vector[Double]): (Double, Double) = {
    // 所有模型参数之和
    // 直接求L2范式的累加和
    /*
        val xAnduSqnorm = admmStates.map(x => {
          DenseVector[Double](norm(x._2.x), norm(x._2.y))
        }
        ).treeReduce(_ + _)

        val r: Double = math.max(xAnduSqnorm(0), norm(admm_z) * math.sqrt(npart))
        val eps_primal = r * 1e-3 + math.sqrt(dim_x * npart) * 1e-3

        val eps_dual: Double = xAnduSqnorm(1) * 1e-3 + math.sqrt(dim_x * npart) * 1e-3
        (eps_primal, eps_dual)
    */


    //先求L2范式的平方在累加求和
    val xAnduSqnorm = admmStates.map(state => {
      DenseVector[Double](math.pow(norm(state.x), 2), math.pow(norm(state.y), 2))
    }
    ).treeReduce(_ + _)

    val r: Double = math.max(math.sqrt(xAnduSqnorm(0)), norm(admm_z) * math.sqrt(npart))
    //TODO
    val eps_primal = r * 1e-3 + math.sqrt(dim_x * npart) * 1e-3

    //TODO
    val eps_dual: Double = math.sqrt(xAnduSqnorm(1)) * 1e-3 + math.sqrt(dim_x * npart) * 1e-3
    (eps_primal, eps_dual)


  }


  def update_rho() {
    /* if (resid_primal  > 10 * resid_dual ) {
      updater.rho *= 2
    } else if (resid_dual  > 10 * resid_primal ) {
      updater.rho /= 2
    }*/

    if (resid_primal / eps_primal > 10 * resid_dual / eps_dual) {
      updater.rho *= 2

    } else if (resid_dual / eps_dual > 10 * resid_primal / eps_primal) {
      updater.rho /= 2

    }
    if (resid_primal < eps_primal) {
      updater.rho /= 1.2
    }

    if (resid_dual < eps_dual) {
      updater.rho *= 1.2
    }
  }


}
