use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };


fn Y_values() -> Tensor<FixedType>  {
    let tensor = TensorTrait::<i32>::new( 
           shape: array![506].span(),
           data: array![ 
               FixedTrait::new(393467, true ),
               FixedTrait::new(224497, true ),
               FixedTrait::new(270821, false ),
               FixedTrait::new(314111, false ),
               FixedTrait::new(541096, false ),
               FixedTrait::new(225687, false ),
               FixedTrait::new(6672, true ),
               FixedTrait::new(495715, false ),
               FixedTrait::new(326130, false ),
               FixedTrait::new(1327, true ),
               FixedTrait::new(262111, true ),
               FixedTrait::new(176081, true ),
               FixedTrait::new(52001, false ),
               FixedTrait::new(55515, false ),
               FixedTrait::new(71007, true ),
               FixedTrait::new(39486, false ),
               FixedTrait::new(168590, false ),
               FixedTrait::new(38574, false ),
               FixedTrait::new(263585, false ),
               FixedTrait::new(13509, true ),
               FixedTrait::new(70526, false ),
               FixedTrait::new(126416, false ),
               FixedTrait::new(41476, true ),
               FixedTrait::new(45463, false ),
               FixedTrait::new(5133, true ),
               FixedTrait::new(33640, false ),
               FixedTrait::new(74450, false ),
               FixedTrait::new(5998, false ),
               FixedTrait::new(75194, true ),
               FixedTrait::new(8098, false ),
               FixedTrait::new(81584, false ),
               FixedTrait::new(233257, true ),
               FixedTrait::new(287633, false ),
               FixedTrait::new(77513, true ),
               FixedTrait::new(13550, true ),
               FixedTrait::new(322085, true ),
               FixedTrait::new(153481, true ),
               FixedTrait::new(138209, true ),
               FixedTrait::new(116980, false ),
               FixedTrait::new(36544, true ),
               FixedTrait::new(44885, false ),
               FixedTrait::new(93098, true ),
               FixedTrait::new(6300, false ),
               FixedTrait::new(5911, false ),
               FixedTrait::new(114130, true ),
               FixedTrait::new(183284, true ),
               FixedTrait::new(27734, true ),
               FixedTrait::new(94145, true ),
               FixedTrait::new(346911, false ),
               FixedTrait::new(143780, false ),
               FixedTrait::new(103646, true ),
               FixedTrait::new(227555, true ),
               FixedTrait::new(174053, true ),
               FixedTrait::new(42534, true ),
               FixedTrait::new(231876, false ),
               FixedTrait::new(278354, false ),
               FixedTrait::new(10280, true ),
               FixedTrait::new(98906, true ),
               FixedTrait::new(99917, false ),
               FixedTrait::new(97316, true ),
               FixedTrait::new(54225, false ),
               FixedTrait::new(164567, true ),
               FixedTrait::new(117140, true ),
               FixedTrait::new(160295, false ),
               FixedTrait::new(630909, false ),
               FixedTrait::new(449674, true ),
               FixedTrait::new(401772, true ),
               FixedTrait::new(58105, false ),
               FixedTrait::new(1411, true ),
               FixedTrait::new(7547, false ),
               FixedTrait::new(65633, true ),
               FixedTrait::new(2795, true ),
               FixedTrait::new(115176, true ),
               FixedTrait::new(42136, true ),
               FixedTrait::new(92077, true ),
               FixedTrait::new(168226, true ),
               FixedTrait::new(193033, true ),
               FixedTrait::new(167575, true ),
               FixedTrait::new(4062, true ),
               FixedTrait::new(139471, true ),
               FixedTrait::new(26592, true ),
               FixedTrait::new(202824, true ),
               FixedTrait::new(80986, true ),
               FixedTrait::new(141474, true ),
               FixedTrait::new(57970, true ),
               FixedTrait::new(78020, true ),
               FixedTrait::new(21722, false ),
               FixedTrait::new(242008, true ),
               FixedTrait::new(463642, true ),
               FixedTrait::new(139664, true ),
               FixedTrait::new(296158, true ),
               FixedTrait::new(354724, true ),
               FixedTrait::new(395917, true ),
               FixedTrait::new(267456, true ),
               FixedTrait::new(422037, true ),
               FixedTrait::new(14719, true ),
               FixedTrait::new(218067, true ),
               FixedTrait::new(191260, false ),
               FixedTrait::new(569209, false ),
               FixedTrait::new(62191, false ),
               FixedTrait::new(191350, false ),
               FixedTrait::new(59366, false ),
               FixedTrait::new(77996, true ),
               FixedTrait::new(66300, true ),
               FixedTrait::new(87479, true ),
               FixedTrait::new(62918, false ),
               FixedTrait::new(151548, false ),
               FixedTrait::new(22969, true ),
               FixedTrait::new(186665, true ),
               FixedTrait::new(24381, true ),
               FixedTrait::new(68835, false ),
               FixedTrait::new(244178, true ),
               FixedTrait::new(129318, true ),
               FixedTrait::new(132086, true ),
               FixedTrait::new(437262, true ),
               FixedTrait::new(139608, true ),
               FixedTrait::new(142688, true ),
               FixedTrait::new(294284, true ),
               FixedTrait::new(4208, false ),
               FixedTrait::new(97767, true ),
               FixedTrait::new(5484, false ),
               FixedTrait::new(142283, true ),
               FixedTrait::new(3760, true ),
               FixedTrait::new(61170, false ),
               FixedTrait::new(115408, true ),
               FixedTrait::new(70895, true ),
               FixedTrait::new(70971, false ),
               FixedTrait::new(66927, false ),
               FixedTrait::new(61517, true ),
               FixedTrait::new(16008, false ),
               FixedTrait::new(54740, true ),
               FixedTrait::new(12443, false ),
               FixedTrait::new(192550, false ),
               FixedTrait::new(173141, false ),
               FixedTrait::new(153586, false ),
               FixedTrait::new(54868, false ),
               FixedTrait::new(99718, false ),
               FixedTrait::new(148218, true ),
               FixedTrait::new(33740, true ),
               FixedTrait::new(88551, false ),
               FixedTrait::new(28087, false ),
               FixedTrait::new(682304, false ),
               FixedTrait::new(78312, true ),
               FixedTrait::new(226176, false ),
               FixedTrait::new(201311, false ),
               FixedTrait::new(115615, false ),
               FixedTrait::new(14471, true ),
               FixedTrait::new(398787, false ),
               FixedTrait::new(529633, false ),
               FixedTrait::new(39025, false ),
               FixedTrait::new(43346, false ),
               FixedTrait::new(85130, false ),
               FixedTrait::new(316068, true ),
               FixedTrait::new(138541, false ),
               FixedTrait::new(351666, true ),
               FixedTrait::new(295158, true ),
               FixedTrait::new(34161, true ),
               FixedTrait::new(526920, false ),
               FixedTrait::new(309996, true ),
               FixedTrait::new(148604, true ),
               FixedTrait::new(374097, true ),
               FixedTrait::new(866735, false ),
               FixedTrait::new(618813, false ),
               FixedTrait::new(534296, false ),
               FixedTrait::new(136883, true ),
               FixedTrait::new(24831, true ),
               FixedTrait::new(838633, false ),
               FixedTrait::new(46695, false ),
               FixedTrait::new(170573, true ),
               FixedTrait::new(285332, true ),
               FixedTrait::new(337847, true ),
               FixedTrait::new(340186, true ),
               FixedTrait::new(8088, false ),
               FixedTrait::new(358609, true ),
               FixedTrait::new(257028, true ),
               FixedTrait::new(86644, true ),
               FixedTrait::new(158380, true ),
               FixedTrait::new(297363, true ),
               FixedTrait::new(100644, true ),
               FixedTrait::new(280342, false ),
               FixedTrait::new(332634, false ),
               FixedTrait::new(552762, false ),
               FixedTrait::new(262938, false ),
               FixedTrait::new(98803, false ),
               FixedTrait::new(241290, false ),
               FixedTrait::new(316769, false ),
               FixedTrait::new(925042, false ),
               FixedTrait::new(93373, true ),
               FixedTrait::new(171179, true ),
               FixedTrait::new(25224, false ),
               FixedTrait::new(408872, false ),
               FixedTrait::new(13805, false ),
               FixedTrait::new(228118, false ),
               FixedTrait::new(66362, true ),
               FixedTrait::new(161134, true ),
               FixedTrait::new(599945, false ),
               FixedTrait::new(185316, true ),
               FixedTrait::new(155268, true ),
               FixedTrait::new(6861, true ),
               FixedTrait::new(315001, false ),
               FixedTrait::new(147853, false ),
               FixedTrait::new(339948, true ),
               FixedTrait::new(342656, false ),
               FixedTrait::new(423891, false ),
               FixedTrait::new(446333, false ),
               FixedTrait::new(5921, true ),
               FixedTrait::new(46999, false ),
               FixedTrait::new(304449, false ),
               FixedTrait::new(59356, false ),
               FixedTrait::new(196222, false ),
               FixedTrait::new(45384, true ),
               FixedTrait::new(146772, false ),
               FixedTrait::new(22212, true ),
               FixedTrait::new(188781, false ),
               FixedTrait::new(824497, false ),
               FixedTrait::new(32080, false ),
               FixedTrait::new(216488, true ),
               FixedTrait::new(22597, false ),
               FixedTrait::new(223822, true ),
               FixedTrait::new(438208, true ),
               FixedTrait::new(424948, true ),
               FixedTrait::new(135958, true ),
               FixedTrait::new(304121, true ),
               FixedTrait::new(23211, false ),
               FixedTrait::new(421329, false ),
               FixedTrait::new(667509, false ),
               FixedTrait::new(913, false ),
               FixedTrait::new(52398, true ),
               FixedTrait::new(736844, false ),
               FixedTrait::new(17425, false ),
               FixedTrait::new(12090, true ),
               FixedTrait::new(104095, true ),
               FixedTrait::new(239330, false ),
               FixedTrait::new(729855, false ),
               FixedTrait::new(177853, true ),
               FixedTrait::new(83037, true ),
               FixedTrait::new(327687, true ),
               FixedTrait::new(79945, true ),
               FixedTrait::new(309799, true ),
               FixedTrait::new(336160, true ),
               FixedTrait::new(346931, true ),
               FixedTrait::new(238723, true ),
               FixedTrait::new(125834, true ),
               FixedTrait::new(242619, true ),
               FixedTrait::new(83324, false ),
               FixedTrait::new(334304, false ),
               FixedTrait::new(280730, false ),
               FixedTrait::new(41822, false ),
               FixedTrait::new(210481, false ),
               FixedTrait::new(138943, false ),
               FixedTrait::new(12691, false ),
               FixedTrait::new(15870, true ),
               FixedTrait::new(306732, false ),
               FixedTrait::new(842423, false ),
               FixedTrait::new(135809, true ),
               FixedTrait::new(52154, true ),
               FixedTrait::new(425257, false ),
               FixedTrait::new(438934, false ),
               FixedTrait::new(31694, true ),
               FixedTrait::new(320463, true ),
               FixedTrait::new(66329, true ),
               FixedTrait::new(388870, false ),
               FixedTrait::new(511883, false ),
               FixedTrait::new(225859, true ),
               FixedTrait::new(43648, false ),
               FixedTrait::new(356892, true ),
               FixedTrait::new(34520, true ),
               FixedTrait::new(600338, false ),
               FixedTrait::new(274076, false ),
               FixedTrait::new(328216, true ),
               FixedTrait::new(78836, true ),
               FixedTrait::new(131295, true ),
               FixedTrait::new(269464, true ),
               FixedTrait::new(18138, true ),
               FixedTrait::new(242902, true ),
               FixedTrait::new(117747, true ),
               FixedTrait::new(157998, true ),
               FixedTrait::new(114028, true ),
               FixedTrait::new(82046, true ),
               FixedTrait::new(13749, true ),
               FixedTrait::new(432696, false ),
               FixedTrait::new(70042, false ),
               FixedTrait::new(370957, false ),
               FixedTrait::new(349104, false ),
               FixedTrait::new(39525, false ),
               FixedTrait::new(351049, true ),
               FixedTrait::new(114, true ),
               FixedTrait::new(251793, true ),
               FixedTrait::new(322020, true ),
               FixedTrait::new(138540, true ),
               FixedTrait::new(323461, true ),
               FixedTrait::new(189825, false ),
               FixedTrait::new(257779, true ),
               FixedTrait::new(125687, true ),
               FixedTrait::new(178901, true ),
               FixedTrait::new(9329, false ),
               FixedTrait::new(17214, true ),
               FixedTrait::new(49854, false ),
               FixedTrait::new(433396, true ),
               FixedTrait::new(190745, true ),
               FixedTrait::new(391354, true ),
               FixedTrait::new(455000, true ),
               FixedTrait::new(162654, true ),
               FixedTrait::new(19734, false ),
               FixedTrait::new(189461, false ),
               FixedTrait::new(155210, true ),
               FixedTrait::new(141706, true ),
               FixedTrait::new(295505, true ),
               FixedTrait::new(382890, true ),
               FixedTrait::new(215591, true ),
               FixedTrait::new(160082, true ),
               FixedTrait::new(313189, true ),
               FixedTrait::new(254367, true ),
               FixedTrait::new(258592, true ),
               FixedTrait::new(110179, true ),
               FixedTrait::new(284367, true ),
               FixedTrait::new(12076, false ),
               FixedTrait::new(93335, false ),
               FixedTrait::new(78033, true ),
               FixedTrait::new(21318, true ),
               FixedTrait::new(71225, true ),
               FixedTrait::new(115957, true ),
               FixedTrait::new(161842, true ),
               FixedTrait::new(62340, true ),
               FixedTrait::new(7722, true ),
               FixedTrait::new(4447, true ),
               FixedTrait::new(44614, true ),
               FixedTrait::new(187374, false ),
               FixedTrait::new(122826, true ),
               FixedTrait::new(108297, true ),
               FixedTrait::new(117480, true ),
               FixedTrait::new(189036, true ),
               FixedTrait::new(258138, true ),
               FixedTrait::new(3892, false ),
               FixedTrait::new(56039, true ),
               FixedTrait::new(31540, false ),
               FixedTrait::new(43317, true ),
               FixedTrait::new(51439, true ),
               FixedTrait::new(102676, true ),
               FixedTrait::new(147433, true ),
               FixedTrait::new(178873, true ),
               FixedTrait::new(155458, false ),
               FixedTrait::new(363551, true ),
               FixedTrait::new(249461, true ),
               FixedTrait::new(173805, false ),
               FixedTrait::new(62586, false ),
               FixedTrait::new(158361, false ),
               FixedTrait::new(142462, true ),
               FixedTrait::new(199363, true ),
               FixedTrait::new(291741, false ),
               FixedTrait::new(159944, false ),
               FixedTrait::new(232911, false ),
               FixedTrait::new(112680, false ),
               FixedTrait::new(307853, false ),
               FixedTrait::new(253960, false ),
               FixedTrait::new(262479, false ),
               FixedTrait::new(120392, true ),
               FixedTrait::new(66719, true ),
               FixedTrait::new(32624, false ),
               FixedTrait::new(222463, false ),
               FixedTrait::new(152950, false ),
               FixedTrait::new(63443, false ),
               FixedTrait::new(168527, false ),
               FixedTrait::new(224887, true ),
               FixedTrait::new(1021999, true ),
               FixedTrait::new(866260, false ),
               FixedTrait::new(416621, false ),
               FixedTrait::new(804020, false ),
               FixedTrait::new(1716995, false ),
               FixedTrait::new(1137437, false ),
               FixedTrait::new(1008806, false ),
               FixedTrait::new(1642115, false ),
               FixedTrait::new(1572876, false ),
               FixedTrait::new(502901, false ),
               FixedTrait::new(853423, false ),
               FixedTrait::new(675488, true ),
               FixedTrait::new(251698, true ),
               FixedTrait::new(454344, true ),
               FixedTrait::new(179130, true ),
               FixedTrait::new(434839, true ),
               FixedTrait::new(260174, true ),
               FixedTrait::new(496555, true ),
               FixedTrait::new(139439, true ),
               FixedTrait::new(49923, true ),
               FixedTrait::new(361812, false ),
               FixedTrait::new(56375, true ),
               FixedTrait::new(286495, false ),
               FixedTrait::new(116742, false ),
               FixedTrait::new(245629, false ),
               FixedTrait::new(177448, true ),
               FixedTrait::new(138428, true ),
               FixedTrait::new(386736, false ),
               FixedTrait::new(12528, true ),
               FixedTrait::new(420822, true ),
               FixedTrait::new(343527, true ),
               FixedTrait::new(472151, true ),
               FixedTrait::new(445355, true ),
               FixedTrait::new(513561, true ),
               FixedTrait::new(101687, true ),
               FixedTrait::new(300821, true ),
               FixedTrait::new(411661, true ),
               FixedTrait::new(695451, true ),
               FixedTrait::new(403784, true ),
               FixedTrait::new(306674, true ),
               FixedTrait::new(73520, false ),
               FixedTrait::new(210446, true ),
               FixedTrait::new(251247, false ),
               FixedTrait::new(518851, false ),
               FixedTrait::new(228880, false ),
               FixedTrait::new(501174, false ),
               FixedTrait::new(14619, true ),
               FixedTrait::new(15681, false ),
               FixedTrait::new(1060469, false ),
               FixedTrait::new(294532, false ),
               FixedTrait::new(739331, false ),
               FixedTrait::new(156222, true ),
               FixedTrait::new(384474, true ),
               FixedTrait::new(229662, false ),
               FixedTrait::new(173814, false ),
               FixedTrait::new(406756, true ),
               FixedTrait::new(190056, true ),
               FixedTrait::new(257181, true ),
               FixedTrait::new(149305, false ),
               FixedTrait::new(14726, false ),
               FixedTrait::new(191769, true ),
               FixedTrait::new(106414, true ),
               FixedTrait::new(402778, true ),
               FixedTrait::new(208089, true ),
               FixedTrait::new(213487, true ),
               FixedTrait::new(232151, true ),
               FixedTrait::new(239832, true ),
               FixedTrait::new(301173, true ),
               FixedTrait::new(355682, true ),
               FixedTrait::new(179006, true ),
               FixedTrait::new(279242, true ),
               FixedTrait::new(2528, false ),
               FixedTrait::new(322489, true ),
               FixedTrait::new(7848, true ),
               FixedTrait::new(231505, false ),
               FixedTrait::new(17426, true ),
               FixedTrait::new(144578, true ),
               FixedTrait::new(12817, true ),
               FixedTrait::new(22314, true ),
               FixedTrait::new(174260, true ),
               FixedTrait::new(46841, true ),
               FixedTrait::new(11403, true ),
               FixedTrait::new(182415, true ),
               FixedTrait::new(362214, true ),
               FixedTrait::new(224024, true ),
               FixedTrait::new(277048, true ),
               FixedTrait::new(204650, true ),
               FixedTrait::new(276097, true ),
               FixedTrait::new(162670, true ),
               FixedTrait::new(307327, true ),
               FixedTrait::new(24903, true ),
               FixedTrait::new(112780, true ),
               FixedTrait::new(835, false ),
               FixedTrait::new(40872, false ),
               FixedTrait::new(149860, true ),
               FixedTrait::new(97486, false ),
               FixedTrait::new(173578, true ),
               FixedTrait::new(162010, true ),
               FixedTrait::new(17961, true ),
               FixedTrait::new(146106, true ),
               FixedTrait::new(70836, false ),
               FixedTrait::new(131978, false ),
               FixedTrait::new(303118, false ),
               FixedTrait::new(141048, false ),
               FixedTrait::new(138670, false ),
               FixedTrait::new(99065, false ),
               FixedTrait::new(17510, true ),
               FixedTrait::new(221326, true ),
               FixedTrait::new(48771, false ),
               FixedTrait::new(276676, false ),
               FixedTrait::new(169834, true ),
               FixedTrait::new(184252, true ),
               FixedTrait::new(251318, true ),
               FixedTrait::new(29967, false ),
               FixedTrait::new(301790, true ),
               FixedTrait::new(30327, true ),
               FixedTrait::new(30722, true ),
               FixedTrait::new(222748, true ),
               FixedTrait::new(233959, true ),
               FixedTrait::new(46924, false ),
               FixedTrait::new(75028, false ),
               FixedTrait::new(66994, true ),
               FixedTrait::new(36432, true ),
               FixedTrait::new(47537, true ),
               FixedTrait::new(219163, false ),
               FixedTrait::new(80207, true ),
               FixedTrait::new(290717, false ),
               FixedTrait::new(10425, true ),
               FixedTrait::new(273207, false ),
               FixedTrait::new(76898, false ),
               FixedTrait::new(254771, false ),
               FixedTrait::new(407278, false ),
               FixedTrait::new(372689, false ),
               FixedTrait::new(52988, true ),
               FixedTrait::new(6425, true ),
               FixedTrait::new(62586, true ),
               FixedTrait::new(240432, true ),
               FixedTrait::new(74274, true ),
               FixedTrait::new(116373, true ),
               FixedTrait::new(244280, true ),
               FixedTrait::new(270530, true ),
               FixedTrait::new(684471, true ),
           ].span(),
            extra: Option::None(())
     ); 

\n }