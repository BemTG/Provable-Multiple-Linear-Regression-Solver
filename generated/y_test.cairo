use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };


fn y_test() -> Tensor<FixedType>  {
    let tensor = TensorTrait::<FixedType>::new( 
           shape: array![51].span(),
           data: array![ 
               FixedTrait::new(13689, false ),
               FixedTrait::new(24175, false ),
               FixedTrait::new(39612, false ),
               FixedTrait::new(23884, false ),
               FixedTrait::new(12961, false ),
               FixedTrait::new(27962, false ),
               FixedTrait::new(22427, false ),
               FixedTrait::new(53448, false ),
               FixedTrait::new(19515, false ),
               FixedTrait::new(7573, false ),
               FixedTrait::new(32768, false ),
               FixedTrait::new(18932, false ),
               FixedTrait::new(40923, false ),
               FixedTrait::new(34078, false ),
               FixedTrait::new(12087, false ),
               FixedTrait::new(5097, false ),
               FixedTrait::new(41069, false ),
               FixedTrait::new(25049, false ),
               FixedTrait::new(14417, false ),
               FixedTrait::new(25049, false ),
               FixedTrait::new(35826, false ),
               FixedTrait::new(21990, false ),
               FixedTrait::new(0, false ),
               FixedTrait::new(24321, false ),
               FixedTrait::new(2912, false ),
               FixedTrait::new(12815, false ),
               FixedTrait::new(10631, false ),
               FixedTrait::new(38739, false ),
               FixedTrait::new(3640, false ),
               FixedTrait::new(65536, false ),
               FixedTrait::new(18058, false ),
               FixedTrait::new(40777, false ),
               FixedTrait::new(41069, false ),
               FixedTrait::new(22864, false ),
               FixedTrait::new(28544, false ),
               FixedTrait::new(56506, false ),
               FixedTrait::new(20534, false ),
               FixedTrait::new(20971, false ),
               FixedTrait::new(43981, false ),
               FixedTrait::new(34515, false ),
               FixedTrait::new(12815, false ),
               FixedTrait::new(18641, false ),
               FixedTrait::new(12379, false ),
               FixedTrait::new(19369, false ),
               FixedTrait::new(25194, false ),
               FixedTrait::new(28398, false ),
               FixedTrait::new(28835, false ),
               FixedTrait::new(24175, false ),
               FixedTrait::new(20971, false ),
               FixedTrait::new(6699, false ),
               FixedTrait::new(56069, false ),
           ].span(),
            extra: Option::None(())
    );
    return tensor; 

 }