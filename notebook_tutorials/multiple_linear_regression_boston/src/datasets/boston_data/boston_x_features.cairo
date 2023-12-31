use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn boston_x_features() ->  Tensor<FP16x16>  {
    let tensor = TensorTrait::<FP16x16>::new( 
    shape: array![50,11].span(),
    data: array![ 
    FixedTrait::new(26719, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(406323, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(33226, false ),
    FixedTrait::new(403963, false ),
    FixedTrait::new(5983436, false ),
    FixedTrait::new(199753, false ),
    FixedTrait::new(524288, false ),
    FixedTrait::new(20119552, false ),
    FixedTrait::new(1140326, false ),
    FixedTrait::new(17588, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(635043, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(38338, false ),
    FixedTrait::new(379715, false ),
    FixedTrait::new(4626841, false ),
    FixedTrait::new(189575, false ),
    FixedTrait::new(393216, false ),
    FixedTrait::new(25624576, false ),
    FixedTrait::new(1258291, false ),
    FixedTrait::new(3512, false ),
    FixedTrait::new(1376256, false ),
    FixedTrait::new(369623, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(28770, false ),
    FixedTrait::new(426704, false ),
    FixedTrait::new(1382809, false ),
    FixedTrait::new(446608, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(15925248, false ),
    FixedTrait::new(1101004, false ),
    FixedTrait::new(731407, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(48496, false ),
    FixedTrait::new(434438, false ),
    FixedTrait::new(6199705, false ),
    FixedTrait::new(139244, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(151643, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1283194, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(39649, false ),
    FixedTrait::new(385351, false ),
    FixedTrait::new(6376652, false ),
    FixedTrait::new(156545, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(26411008, false ),
    FixedTrait::new(963379, false ),
    FixedTrait::new(637283, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(48496, false ),
    FixedTrait::new(419823, false ),
    FixedTrait::new(6370099, false ),
    FixedTrait::new(135338, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(6860, false ),
    FixedTrait::new(2621440, false ),
    FixedTrait::new(420085, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(29294, false ),
    FixedTrait::new(476250, false ),
    FixedTrait::new(3211264, false ),
    FixedTrait::new(313733, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(16646144, false ),
    FixedTrait::new(1153433, false ),
    FixedTrait::new(1598672, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(45875, false ),
    FixedTrait::new(304873, false ),
    FixedTrait::new(6553600, false ),
    FixedTrait::new(96154, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(264661, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(34865, false ),
    FixedTrait::new(408223, false ),
    FixedTrait::new(5944115, false ),
    FixedTrait::new(203115, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(10624, false ),
    FixedTrait::new(1310720, false ),
    FixedTrait::new(456130, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(30408, false ),
    FixedTrait::new(408944, false ),
    FixedTrait::new(1068236, false ),
    FixedTrait::new(290258, false ),
    FixedTrait::new(196608, false ),
    FixedTrait::new(14614528, false ),
    FixedTrait::new(1218969, false ),
    FixedTrait::new(6768, false ),
    FixedTrait::new(1638400, false ),
    FixedTrait::new(336199, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(29687, false ),
    FixedTrait::new(388431, false ),
    FixedTrait::new(3093299, false ),
    FixedTrait::new(454295, false ),
    FixedTrait::new(524288, false ),
    FixedTrait::new(18612224, false ),
    FixedTrait::new(1291059, false ),
    FixedTrait::new(40077, false ),
    FixedTrait::new(1310720, false ),
    FixedTrait::new(260177, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(42401, false ),
    FixedTrait::new(570425, false ),
    FixedTrait::new(5695078, false ),
    FixedTrait::new(118030, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(17301504, false ),
    FixedTrait::new(851968, false ),
    FixedTrait::new(527944, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(38273, false ),
    FixedTrait::new(355663, false ),
    FixedTrait::new(6252134, false ),
    FixedTrait::new(159239, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(14030, false ),
    FixedTrait::new(1441792, false ),
    FixedTrait::new(384040, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(28246, false ),
    FixedTrait::new(421920, false ),
    FixedTrait::new(583270, false ),
    FixedTrait::new(484750, false ),
    FixedTrait::new(458752, false ),
    FixedTrait::new(21626880, false ),
    FixedTrait::new(1251737, false ),
    FixedTrait::new(22353, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(483655, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(32309, false ),
    FixedTrait::new(420413, false ),
    FixedTrait::new(2627993, false ),
    FixedTrait::new(309402, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(18808832, false ),
    FixedTrait::new(1284505, false ),
    FixedTrait::new(51800, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(648806, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35651, false ),
    FixedTrait::new(401211, false ),
    FixedTrait::new(3460300, false ),
    FixedTrait::new(173034, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(19922944, false ),
    FixedTrait::new(1205862, false ),
    FixedTrait::new(1998, false ),
    FixedTrait::new(3604480, false ),
    FixedTrait::new(247726, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(31719, false ),
    FixedTrait::new(450494, false ),
    FixedTrait::new(1841561, false ),
    FixedTrait::new(423716, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(24248320, false ),
    FixedTrait::new(1153433, false ),
    FixedTrait::new(22898, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(648806, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35651, false ),
    FixedTrait::new(391380, false ),
    FixedTrait::new(5026611, false ),
    FixedTrait::new(203325, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(19922944, false ),
    FixedTrait::new(1205862, false ),
    FixedTrait::new(24195, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(648806, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35651, false ),
    FixedTrait::new(430374, false ),
    FixedTrait::new(5721292, false ),
    FixedTrait::new(236080, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(19922944, false ),
    FixedTrait::new(1205862, false ),
    FixedTrait::new(623485, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(46727, false ),
    FixedTrait::new(440926, false ),
    FixedTrait::new(6166937, false ),
    FixedTrait::new(163584, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(52606, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(533463, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35258, false ),
    FixedTrait::new(357564, false ),
    FixedTrait::new(2398617, false ),
    FixedTrait::new(248807, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(20119552, false ),
    FixedTrait::new(1376256, false ),
    FixedTrait::new(3709, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(223477, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(32047, false ),
    FixedTrait::new(459210, false ),
    FixedTrait::new(5655756, false ),
    FixedTrait::new(224244, false ),
    FixedTrait::new(131072, false ),
    FixedTrait::new(17694720, false ),
    FixedTrait::new(1166540, false ),
    FixedTrait::new(28554, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(694026, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(32047, false ),
    FixedTrait::new(350224, false ),
    FixedTrait::new(6553600, false ),
    FixedTrait::new(253952, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(18153472, false ),
    FixedTrait::new(1218969, false ),
    FixedTrait::new(34087, false ),
    FixedTrait::new(1310720, false ),
    FixedTrait::new(260177, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(42401, false ),
    FixedTrait::new(550371, false ),
    FixedTrait::new(5996544, false ),
    FixedTrait::new(149979, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(17301504, false ),
    FixedTrait::new(851968, false ),
    FixedTrait::new(802632, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(38273, false ),
    FixedTrait::new(382533, false ),
    FixedTrait::new(3912499, false ),
    FixedTrait::new(130914, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(17654, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(648806, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35651, false ),
    FixedTrait::new(410648, false ),
    FixedTrait::new(5426380, false ),
    FixedTrait::new(213830, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(19922944, false ),
    FixedTrait::new(1205862, false ),
    FixedTrait::new(2988, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(910295, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(36044, false ),
    FixedTrait::new(385875, false ),
    FixedTrait::new(3670016, false ),
    FixedTrait::new(203954, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(18087936, false ),
    FixedTrait::new(1074790, false ),
    FixedTrait::new(3787, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(161218, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(31981, false ),
    FixedTrait::new(457441, false ),
    FixedTrait::new(3827302, false ),
    FixedTrait::new(185401, false ),
    FixedTrait::new(196608, false ),
    FixedTrait::new(12648448, false ),
    FixedTrait::new(1166540, false ),
    FixedTrait::new(54084, false ),
    FixedTrait::new(1310720, false ),
    FixedTrait::new(260177, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(42401, false ),
    FixedTrait::new(480182, false ),
    FixedTrait::new(6193152, false ),
    FixedTrait::new(136236, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(17301504, false ),
    FixedTrait::new(851968, false ),
    FixedTrait::new(227690, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(47054, false ),
    FixedTrait::new(575406, false ),
    FixedTrait::new(5432934, false ),
    FixedTrait::new(124826, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(35685, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1434583, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(40894, false ),
    FixedTrait::new(403111, false ),
    FixedTrait::new(6415974, false ),
    FixedTrait::new(109359, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(28639232, false ),
    FixedTrait::new(1389363, false ),
    FixedTrait::new(9209, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(694026, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(32047, false ),
    FixedTrait::new(417792, false ),
    FixedTrait::new(2116812, false ),
    FixedTrait::new(258565, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(18153472, false ),
    FixedTrait::new(1218969, false ),
    FixedTrait::new(3069, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(223477, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(32047, false ),
    FixedTrait::new(420544, false ),
    FixedTrait::new(4331929, false ),
    FixedTrait::new(202656, false ),
    FixedTrait::new(131072, false ),
    FixedTrait::new(17694720, false ),
    FixedTrait::new(1166540, false ),
    FixedTrait::new(4016, false ),
    FixedTrait::new(1310720, false ),
    FixedTrait::new(218234, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(29025, false ),
    FixedTrait::new(501022, false ),
    FixedTrait::new(3257139, false ),
    FixedTrait::new(341567, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(14155776, false ),
    FixedTrait::new(976486, false ),
    FixedTrait::new(63974, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1434583, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(40894, false ),
    FixedTrait::new(377290, false ),
    FixedTrait::new(6448742, false ),
    FixedTrait::new(153747, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(28639232, false ),
    FixedTrait::new(1389363, false ),
    FixedTrait::new(14556, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(656015, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35848, false ),
    FixedTrait::new(399245, false ),
    FixedTrait::new(6252134, false ),
    FixedTrait::new(166985, false ),
    FixedTrait::new(393216, false ),
    FixedTrait::new(28311552, false ),
    FixedTrait::new(1166540, false ),
    FixedTrait::new(11358, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(635043, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(38338, false ),
    FixedTrait::new(374013, false ),
    FixedTrait::new(3538944, false ),
    FixedTrait::new(156087, false ),
    FixedTrait::new(393216, false ),
    FixedTrait::new(25624576, false ),
    FixedTrait::new(1258291, false ),
    FixedTrait::new(3758, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(294256, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(29425, false ),
    FixedTrait::new(434503, false ),
    FixedTrait::new(3676569, false ),
    FixedTrait::new(290829, false ),
    FixedTrait::new(196608, false ),
    FixedTrait::new(16187392, false ),
    FixedTrait::new(1212416, false ),
    FixedTrait::new(11143, false ),
    FixedTrait::new(819200, false ),
    FixedTrait::new(515768, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(34340, false ),
    FixedTrait::new(393478, false ),
    FixedTrait::new(5629542, false ),
    FixedTrait::new(432019, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(20381696, false ),
    FixedTrait::new(996147, false ),
    FixedTrait::new(2121, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(142868, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(30015, false ),
    FixedTrait::new(458620, false ),
    FixedTrait::new(3001548, false ),
    FixedTrait::new(397292, false ),
    FixedTrait::new(196608, false ),
    FixedTrait::new(14548992, false ),
    FixedTrait::new(1225523, false ),
    FixedTrait::new(10443, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(452853, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(29360, false ),
    FixedTrait::new(407044, false ),
    FixedTrait::new(425984, false ),
    FixedTrait::new(374924, false ),
    FixedTrait::new(196608, false ),
    FixedTrait::new(15269888, false ),
    FixedTrait::new(1173094, false ),
    FixedTrait::new(3057, false ),
    FixedTrait::new(5242880, false ),
    FixedTrait::new(99614, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(26476, false ),
    FixedTrait::new(465764, false ),
    FixedTrait::new(2398617, false ),
    FixedTrait::new(479002, false ),
    FixedTrait::new(131072, false ),
    FixedTrait::new(21561344, false ),
    FixedTrait::new(825753, false ),
    FixedTrait::new(2325, false ),
    FixedTrait::new(5242880, false ),
    FixedTrait::new(238551, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(25690, false ),
    FixedTrait::new(385089, false ),
    FixedTrait::new(1251737, false ),
    FixedTrait::new(604261, false ),
    FixedTrait::new(65536, false ),
    FixedTrait::new(20643840, false ),
    FixedTrait::new(1074790, false ),
    FixedTrait::new(4601, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(265420, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(33423, false ),
    FixedTrait::new(394526, false ),
    FixedTrait::new(3093299, false ),
    FixedTrait::new(232973, false ),
    FixedTrait::new(327680, false ),
    FixedTrait::new(19398656, false ),
    FixedTrait::new(1087897, false ),
    FixedTrait::new(2816, false ),
    FixedTrait::new(3440640, false ),
    FixedTrait::new(348651, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(26542, false ),
    FixedTrait::new(430243, false ),
    FixedTrait::new(1500774, false ),
    FixedTrait::new(479540, false ),
    FixedTrait::new(393216, false ),
    FixedTrait::new(19202048, false ),
    FixedTrait::new(1087897, false ),
    FixedTrait::new(90963, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(533463, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(35258, false ),
    FixedTrait::new(389939, false ),
    FixedTrait::new(5373952, false ),
    FixedTrait::new(261488, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(20119552, false ),
    FixedTrait::new(1376256, false ),
    FixedTrait::new(12499, false ),
    FixedTrait::new(1441792, false ),
    FixedTrait::new(384040, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(28246, false ),
    FixedTrait::new(440270, false ),
    FixedTrait::new(1146880, false ),
    FixedTrait::new(512917, false ),
    FixedTrait::new(458752, false ),
    FixedTrait::new(21626880, false ),
    FixedTrait::new(1251737, false ),
    FixedTrait::new(12805, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(708444, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(27066, false ),
    FixedTrait::new(409272, false ),
    FixedTrait::new(406323, false ),
    FixedTrait::new(346508, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(19988480, false ),
    FixedTrait::new(1258291, false ),
    FixedTrait::new(758769, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(1186201, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(45875, false ),
    FixedTrait::new(330039, false ),
    FixedTrait::new(6356992, false ),
    FixedTrait::new(115998, false ),
    FixedTrait::new(1572864, false ),
    FixedTrait::new(43646976, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(2369, false ),
    FixedTrait::new(5242880, false ),
    FixedTrait::new(324403, false ),
    FixedTrait::new(0, false ),
    FixedTrait::new(26935, false ),
    FixedTrait::new(434503, false ),
    FixedTrait::new(1533542, false ),
    FixedTrait::new(335328, false ),
    FixedTrait::new(262144, false ),
    FixedTrait::new(16056320, false ),
    FixedTrait::new(1258291, false ),
].span() 
 
);

return tensor; 
}