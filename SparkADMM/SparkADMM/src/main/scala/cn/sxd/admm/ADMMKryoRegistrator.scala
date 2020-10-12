package cn.sxd.admm

import breeze.linalg.{DenseMatrix, DenseVector, SparseVector}
import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator

class ADMMKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[DenseVector[Double]])
    kryo.register(classOf[DenseMatrix[Double]])
    kryo.register(classOf[SparseVector[Double]])
    kryo.register(classOf[ADMMState])
  }
}
