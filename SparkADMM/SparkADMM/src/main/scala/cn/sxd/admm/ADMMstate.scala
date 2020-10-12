package cn.sxd.admm

//import org.apache.spark.mllib.linalg.DenseVector
//Breeze :是机器学习和数值技术库 ，它是sparkMlib的核心，包括线性代数、数值技术和优化，是一种通用、功能强大、有效的机器学习方法。线性学习计算库
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.regression.LabeledPoint

/*case class ADMMState(var x:DenseVector[Double],
                     var z:DenseVector[Double],u:DenseVector[Double],var z_pre:DenseVector[Double],var w:DenseVector[Double])*/
case class ADMMState(var x:BDV[Double], var y:BDV[Double])
object ADMMState {
  //初始化x,y,u,z_pre,w向量，长度为训练数据的列数
  def apply(initialWeight:Array[Double]):ADMMState={
    new ADMMState(
      x = BDV(initialWeight),
      y = zeros(initialWeight.length)
    )
  }

  //初始化各个向量：全部设置为0向量
  def zeros(n:Int):BDV[Double]={
    BDV.zeros(n)
  }
/*
  def zeros(n:Int):BSV[Double]={
    BSV.zeros(n)
  }*/
}
