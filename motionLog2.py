from frame2d import Frame2D
robotFrames = [
   (0, Frame2D.fromXYA(0.000000,0.000000,-0.000988)),
   (1, Frame2D.fromXYA(0.000000,0.000000,-0.000953)),
   (2, Frame2D.fromXYA(0.000000,0.000000,-0.000947)),
   (3, Frame2D.fromXYA(0.000000,0.000000,-0.000921)),
   (4, Frame2D.fromXYA(0.000000,0.000000,-0.000845)),
   (5, Frame2D.fromXYA(0.000000,0.000000,-0.000813)),
   (6, Frame2D.fromXYA(0.000000,0.000000,-0.000770)),
   (7, Frame2D.fromXYA(0.000000,0.000000,-0.000737)),
   (8, Frame2D.fromXYA(0.000000,0.000000,-0.000688)),
   (9, Frame2D.fromXYA(0.129763,0.000957,-0.000530)),
   (10, Frame2D.fromXYA(0.875418,0.211101,0.019539)),
   (11, Frame2D.fromXYA(1.744904,0.507955,0.036166)),
   (12, Frame2D.fromXYA(2.733109,0.935285,0.057607)),
   (13, Frame2D.fromXYA(4.189283,1.626733,0.089350)),
   (14, Frame2D.fromXYA(6.538312,1.986997,0.102548)),
   (15, Frame2D.fromXYA(7.596784,2.207114,0.109662)),
   (16, Frame2D.fromXYA(9.298397,2.747268,0.132336)),
   (17, Frame2D.fromXYA(11.033090,3.434323,0.157989)),
   (18, Frame2D.fromXYA(12.049276,3.790034,0.170290)),
   (19, Frame2D.fromXYA(13.972511,4.428163,0.190477)),
   (20, Frame2D.fromXYA(14.990725,4.785154,0.201645)),
   (21, Frame2D.fromXYA(16.928429,5.604358,0.226406)),
   (22, Frame2D.fromXYA(18.460234,6.218555,0.242008)),
   (23, Frame2D.fromXYA(20.766106,7.291332,0.270522)),
   (24, Frame2D.fromXYA(22.724529,8.207598,0.292793)),
   (25, Frame2D.fromXYA(23.758804,8.660312,0.303909)),
   (26, Frame2D.fromXYA(25.920570,9.732131,0.327276)),
   (27, Frame2D.fromXYA(27.612709,10.601635,0.345627)),
   (28, Frame2D.fromXYA(29.161036,11.451168,0.363484)),
   (29, Frame2D.fromXYA(31.408588,12.596387,0.382115)),
   (30, Frame2D.fromXYA(33.500744,13.765144,0.404242)),
   (31, Frame2D.fromXYA(34.552513,14.350510,0.414359)),
   (32, Frame2D.fromXYA(35.633999,14.978644,0.425487)),
   (33, Frame2D.fromXYA(38.945511,16.937006,0.452791)),
   (34, Frame2D.fromXYA(40.025326,17.576189,0.461341)),
   (35, Frame2D.fromXYA(42.106400,18.870289,0.477440)),
   (36, Frame2D.fromXYA(44.134857,20.272598,0.496107)),
   (37, Frame2D.fromXYA(45.604275,21.227135,0.505265)),
   (38, Frame2D.fromXYA(46.967415,22.134512,0.515434)),
   (39, Frame2D.fromXYA(48.712769,23.383781,0.530333)),
   (40, Frame2D.fromXYA(49.624535,24.045292,0.539379)),
   (41, Frame2D.fromXYA(52.094166,25.783798,0.554572)),
   (42, Frame2D.fromXYA(53.922993,27.103863,0.565641)),
   (43, Frame2D.fromXYA(54.434605,27.479382,0.568765)),
   (44, Frame2D.fromXYA(56.464760,28.836109,0.572216)),
   (45, Frame2D.fromXYA(58.524635,30.272396,0.581701)),
   (46, Frame2D.fromXYA(61.218937,32.104160,0.587240)),
   (47, Frame2D.fromXYA(61.783310,32.495365,0.588538)),
   (48, Frame2D.fromXYA(63.977097,34.072838,0.596921)),
   (49, Frame2D.fromXYA(65.973137,35.917995,0.622071)),
   (50, Frame2D.fromXYA(66.367340,36.336449,0.628647)),
   (51, Frame2D.fromXYA(68.054710,37.954567,0.649313)),
   (52, Frame2D.fromXYA(69.757820,39.543427,0.666575)),
   (53, Frame2D.fromXYA(70.965469,40.842636,0.683345)),
   (54, Frame2D.fromXYA(72.670815,42.705490,0.703124)),
   (55, Frame2D.fromXYA(74.022018,44.024853,0.714098)),
   (56, Frame2D.fromXYA(74.889778,44.933289,0.722823)),
   (57, Frame2D.fromXYA(76.969933,47.218163,0.745341)),
   (58, Frame2D.fromXYA(78.776192,49.140408,0.760347)),
   (59, Frame2D.fromXYA(80.306320,50.982029,0.779917)),
   (60, Frame2D.fromXYA(81.082909,51.974995,0.790282)),
   (61, Frame2D.fromXYA(82.261627,53.372730,0.801695)),
   (62, Frame2D.fromXYA(84.130417,55.742668,0.823450)),
   (63, Frame2D.fromXYA(84.701157,56.668407,0.835726)),
   (64, Frame2D.fromXYA(85.650879,58.414494,0.862240)),
   (65, Frame2D.fromXYA(86.795845,60.321709,0.885000)),
   (66, Frame2D.fromXYA(87.401108,61.284370,0.895138)),
   (67, Frame2D.fromXYA(88.335907,62.704407,0.907438)),
   (68, Frame2D.fromXYA(89.810379,65.130241,0.930867)),
   (69, Frame2D.fromXYA(90.351738,66.124374,0.941059)),
   (70, Frame2D.fromXYA(91.577538,68.108841,0.956548)),
   (71, Frame2D.fromXYA(92.618507,70.051231,0.973289)),
   (72, Frame2D.fromXYA(92.900360,70.542305,0.976095)),
   (73, Frame2D.fromXYA(94.379677,73.168442,0.993234)),
   (74, Frame2D.fromXYA(95.616104,75.348564,1.004589)),
   (75, Frame2D.fromXYA(96.201988,76.395218,1.010085)),
   (76, Frame2D.fromXYA(97.439064,78.572563,1.019820)),
   (77, Frame2D.fromXYA(98.223595,80.086777,1.028464)),
   (78, Frame2D.fromXYA(98.737564,81.166031,1.036432)),
   (79, Frame2D.fromXYA(99.822044,83.579262,1.057242)),
   (80, Frame2D.fromXYA(100.453880,85.443054,1.079010)),
   (81, Frame2D.fromXYA(100.654373,86.337662,1.092093)),
   (82, Frame2D.fromXYA(100.868828,88.046341,1.123324)),
   (83, Frame2D.fromXYA(101.130074,89.515274,1.146425)),
   (84, Frame2D.fromXYA(101.881371,92.192360,1.173974)),
   (85, Frame2D.fromXYA(102.212852,93.277260,1.183304)),
   (86, Frame2D.fromXYA(102.675110,95.383636,1.205041)),
   (87, Frame2D.fromXYA(103.079163,97.576927,1.227878)),
   (88, Frame2D.fromXYA(103.363235,98.738655,1.235785)),
   (89, Frame2D.fromXYA(103.835419,100.954819,1.254426)),
   (90, Frame2D.fromXYA(104.327194,103.167953,1.271475)),
   (91, Frame2D.fromXYA(104.658745,104.964478,1.284343)),
   (92, Frame2D.fromXYA(104.887115,106.668594,1.300363)),
   (93, Frame2D.fromXYA(105.173744,109.055397,1.321643)),
   (94, Frame2D.fromXYA(105.324661,110.181488,1.331203)),
   (95, Frame2D.fromXYA(105.527115,112.507973,1.352192)),
   (96, Frame2D.fromXYA(105.776443,115.408997,1.378272)),
   (97, Frame2D.fromXYA(105.854706,116.023201,1.382157)),
   (98, Frame2D.fromXYA(106.144890,118.653763,1.398407)),
   (99, Frame2D.fromXYA(106.215118,121.177521,1.419671)),
   (100, Frame2D.fromXYA(106.303925,123.569717,1.436179)),
   (101, Frame2D.fromXYA(106.374458,124.830170,1.444292)),
   (102, Frame2D.fromXYA(106.368271,127.733772,1.468489)),
   (103, Frame2D.fromXYA(106.331451,128.304733,1.474152)),
   (104, Frame2D.fromXYA(106.376747,130.690628,1.488156)),
   (105, Frame2D.fromXYA(106.417740,133.074951,1.500988)),
   (106, Frame2D.fromXYA(106.423805,135.650818,1.513585)),
   (107, Frame2D.fromXYA(106.376572,137.410568,1.522610)),
   (108, Frame2D.fromXYA(106.295189,139.170456,1.532706)),
   (109, Frame2D.fromXYA(106.212296,141.432571,1.543511)),
   (110, Frame2D.fromXYA(106.081993,142.938889,1.553880)),
   (111, Frame2D.fromXYA(105.878746,144.505112,1.568099)),
   (112, Frame2D.fromXYA(105.674141,147.388199,1.582444)),
   (113, Frame2D.fromXYA(105.523430,148.583694,1.591296)),
   (114, Frame2D.fromXYA(105.246758,150.957153,1.606933)),
   (115, Frame2D.fromXYA(105.087891,152.141876,1.614483)),
   (116, Frame2D.fromXYA(104.572441,154.810455,1.638817)),
   (117, Frame2D.fromXYA(104.318893,155.862625,1.650534)),
   (118, Frame2D.fromXYA(103.808464,158.712097,1.671285)),
   (119, Frame2D.fromXYA(103.375671,161.182312,1.684603)),
   (120, Frame2D.fromXYA(103.088074,162.285156,1.695034)),
   (121, Frame2D.fromXYA(102.444664,164.792542,1.716007)),
   (122, Frame2D.fromXYA(101.983398,166.682541,1.729174)),
   (123, Frame2D.fromXYA(101.454308,168.442764,1.743991)),
   (124, Frame2D.fromXYA(100.806007,170.743652,1.763964)),
   (125, Frame2D.fromXYA(100.095398,173.097672,1.783033)),
   (126, Frame2D.fromXYA(99.771271,174.237274,1.791686)),
   (127, Frame2D.fromXYA(99.206505,175.970551,1.802764)),
   (128, Frame2D.fromXYA(98.050766,178.859085,1.827321)),
   (129, Frame2D.fromXYA(97.600449,179.899429,1.838331)),
   (130, Frame2D.fromXYA(96.644974,182.038727,1.859462)),
   (131, Frame2D.fromXYA(95.645508,184.029205,1.882948)),
   (132, Frame2D.fromXYA(95.151733,185.111755,1.894182)),
   (133, Frame2D.fromXYA(94.424904,186.653214,1.906727)),
   (134, Frame2D.fromXYA(93.226021,189.489746,1.922612)),
   (135, Frame2D.fromXYA(92.161438,191.901810,1.933948)),
   (136, Frame2D.fromXYA(91.647392,193.052948,1.940897)),
   (137, Frame2D.fromXYA(90.620148,195.140488,1.955093)),
   (138, Frame2D.fromXYA(89.869057,196.662994,1.965622)),
   (139, Frame2D.fromXYA(89.117775,198.176041,1.976279)),
   (140, Frame2D.fromXYA(88.090874,200.058533,1.990825)),
   (141, Frame2D.fromXYA(87.005966,202.113220,2.002398)),
   (142, Frame2D.fromXYA(86.481651,203.173798,2.004953)),
   (143, Frame2D.fromXYA(85.333054,205.128616,2.020284)),
   (144, Frame2D.fromXYA(84.119873,207.048355,2.036699)),
   (145, Frame2D.fromXYA(83.510338,208.005341,2.044801)),
   (146, Frame2D.fromXYA(82.242050,209.890121,2.063459)),
   (147, Frame2D.fromXYA(80.930229,211.816193,2.080427)),
   (148, Frame2D.fromXYA(80.310753,212.763123,2.088840)),
   (149, Frame2D.fromXYA(78.674751,214.917435,2.112019)),
   (150, Frame2D.fromXYA(77.611099,216.108826,2.129002)),
   (151, Frame2D.fromXYA(76.120712,217.836304,2.150140)),
   (152, Frame2D.fromXYA(75.050644,219.002625,2.167134)),
   (153, Frame2D.fromXYA(73.897110,220.115067,2.187323)),
   (154, Frame2D.fromXYA(71.915939,221.850708,2.222400)),
   (155, Frame2D.fromXYA(71.515228,222.245148,2.228276)),
   (156, Frame2D.fromXYA(69.995819,223.923676,2.242293)),
   (157, Frame2D.fromXYA(67.931778,226.050705,2.263284)),
   (158, Frame2D.fromXYA(67.511566,226.425461,2.269477)),
   (159, Frame2D.fromXYA(65.772705,227.908249,2.294460)),
   (160, Frame2D.fromXYA(63.997662,229.347504,2.319095)),
   (161, Frame2D.fromXYA(63.121063,230.062744,2.330329)),
   (162, Frame2D.fromXYA(61.324745,231.477280,2.350781)),
   (163, Frame2D.fromXYA(59.414928,233.019073,2.369494)),
   (164, Frame2D.fromXYA(58.040131,234.120880,2.383604)),
   (165, Frame2D.fromXYA(56.121414,235.552704,2.402682)),
   (166, Frame2D.fromXYA(54.611553,236.691849,2.416029)),
   (167, Frame2D.fromXYA(53.670380,237.412537,2.424118)),
   (168, Frame2D.fromXYA(51.195168,239.154938,2.444438)),
   (169, Frame2D.fromXYA(49.171291,240.453369,2.463980)),
   (170, Frame2D.fromXYA(47.304363,241.622116,2.481936)),
   (171, Frame2D.fromXYA(46.341530,242.209915,2.490982)),
   (172, Frame2D.fromXYA(44.879898,243.084366,2.502661)),
   (173, Frame2D.fromXYA(42.396828,244.585953,2.523016)),
   (174, Frame2D.fromXYA(41.388710,245.208374,2.530339)),
   (175, Frame2D.fromXYA(39.363190,246.478195,2.537752)),
   (176, Frame2D.fromXYA(37.281910,247.759445,2.547829)),
   (177, Frame2D.fromXYA(36.753403,248.094772,2.549621)),
   (178, Frame2D.fromXYA(34.611481,249.406708,2.558430)),
   (179, Frame2D.fromXYA(31.871315,251.056839,2.570182)),
   (180, Frame2D.fromXYA(30.883839,251.608337,2.574791)),
   (181, Frame2D.fromXYA(29.063766,252.365204,2.595394)),
   (182, Frame2D.fromXYA(27.521332,253.085922,2.609407)),
   (183, Frame2D.fromXYA(26.398785,253.629623,2.617600)),
   (184, Frame2D.fromXYA(23.639883,255.013718,2.632003)),
   (185, Frame2D.fromXYA(21.424469,256.051453,2.645943)),
   (186, Frame2D.fromXYA(20.342226,256.535614,2.652281)),
   (187, Frame2D.fromXYA(18.051605,257.439148,2.670795)),
   (188, Frame2D.fromXYA(16.359612,258.115265,2.684904)),
   (189, Frame2D.fromXYA(15.244825,258.561890,2.693462)),
   (190, Frame2D.fromXYA(12.279778,259.749969,2.710889)),
   (191, Frame2D.fromXYA(9.808556,260.691895,2.723643)),
   (192, Frame2D.fromXYA(7.385967,261.548798,2.738396)),
   (193, Frame2D.fromXYA(6.199053,261.954559,2.745930)),
   (194, Frame2D.fromXYA(3.859129,262.537262,2.767479)),
   (195, Frame2D.fromXYA(1.568817,263.038300,2.790468)),
   (196, Frame2D.fromXYA(0.404909,263.318878,2.800397)),
   (197, Frame2D.fromXYA(-1.922314,263.871643,2.819279)),
   (198, Frame2D.fromXYA(-3.146110,264.179871,2.827063)),
   (199, Frame2D.fromXYA(-4.926953,264.571625,2.840842)),
   (200, Frame2D.fromXYA(-7.835439,265.122498,2.863705)),
   (201, Frame2D.fromXYA(-10.250086,265.550049,2.881231)),
   (202, Frame2D.fromXYA(-11.554509,265.778992,2.889433)),
   (203, Frame2D.fromXYA(-13.907027,266.183563,2.904261)),
   (204, Frame2D.fromXYA(-16.045057,266.388092,2.922093)),
   (205, Frame2D.fromXYA(-18.280251,266.735840,2.935598)),
   (206, Frame2D.fromXYA(-19.463032,266.893250,2.941855)),
   (207, Frame2D.fromXYA(-21.307241,266.916992,2.962289)),
   (208, Frame2D.fromXYA(-23.223877,266.887909,2.982295)),
   (209, Frame2D.fromXYA(-24.306063,266.880402,2.992671)),
   (210, Frame2D.fromXYA(-25.425098,266.978424,2.997788)),
   (211, Frame2D.fromXYA(-28.879028,267.226074,3.016336)),
   (212, Frame2D.fromXYA(-29.946602,267.232239,3.024826)),
   (213, Frame2D.fromXYA(-31.964054,267.238708,3.041087)),
   (214, Frame2D.fromXYA(-33.931488,267.082062,3.063662)),
   (215, Frame2D.fromXYA(-34.932632,266.990906,3.074552)),
   (216, Frame2D.fromXYA(-36.905529,266.682556,3.098377)),
   (217, Frame2D.fromXYA(-39.288712,266.405334,3.118758)),
   (218, Frame2D.fromXYA(-41.053764,266.255463,3.132359)),
   (219, Frame2D.fromXYA(-42.931015,266.079346,-3.137143)),
   (220, Frame2D.fromXYA(-45.376507,265.825073,-3.119403)),
   (221, Frame2D.fromXYA(-46.931656,265.521667,-3.101852)),
   (222, Frame2D.fromXYA(-48.476784,265.106171,-3.080164)),
   (223, Frame2D.fromXYA(-51.142082,264.530640,-3.053446)),
   (224, Frame2D.fromXYA(-53.467934,263.946777,-3.029038)),
   (225, Frame2D.fromXYA(-54.633759,263.606384,-3.016986)),
   (226, Frame2D.fromXYA(-57.007610,262.948059,-2.993634)),
   (227, Frame2D.fromXYA(-58.771820,262.476318,-2.980305)),
   (228, Frame2D.fromXYA(-59.998405,262.178345,-2.972170)),
   (229, Frame2D.fromXYA(-62.930138,261.455536,-2.953991)),
   (230, Frame2D.fromXYA(-65.356651,260.826141,-2.940201)),
   (231, Frame2D.fromXYA(-65.889458,260.652527,-2.935469)),
   (232, Frame2D.fromXYA(-68.363770,259.895447,-2.920141)),
   (233, Frame2D.fromXYA(-70.699547,259.161896,-2.905422)),
   (234, Frame2D.fromXYA(-72.552383,258.581665,-2.895637)),
   (235, Frame2D.fromXYA(-75.009590,257.812988,-2.884359)),
   (236, Frame2D.fromXYA(-77.337021,257.037262,-2.872762)),
   (237, Frame2D.fromXYA(-78.529045,256.653870,-2.868473)),
   (238, Frame2D.fromXYA(-80.261093,256.078461,-2.861251)),
   (239, Frame2D.fromXYA(-82.998863,254.982239,-2.839673)),
   (240, Frame2D.fromXYA(-85.357529,254.116821,-2.827917)),
   (241, Frame2D.fromXYA(-86.467323,253.681137,-2.822788)),
   (242, Frame2D.fromXYA(-88.806427,252.762283,-2.811883)),
   (243, Frame2D.fromXYA(-90.503639,252.114273,-2.806265)),
   (244, Frame2D.fromXYA(-91.361908,251.726227,-2.799350)),
   (245, Frame2D.fromXYA(-93.979065,250.485367,-2.781234)),
   (246, Frame2D.fromXYA(-96.213821,249.473068,-2.770656)),
   (247, Frame2D.fromXYA(-97.200363,248.896774,-2.761168)),
   (248, Frame2D.fromXYA(-99.107697,247.779007,-2.740375)),
   (249, Frame2D.fromXYA(-100.898582,246.864670,-2.729571)),
   (250, Frame2D.fromXYA(-102.515953,246.032913,-2.719938)),
   (251, Frame2D.fromXYA(-104.528954,244.858841,-2.702875)),
   (252, Frame2D.fromXYA(-106.320244,243.654694,-2.684207)),
   (253, Frame2D.fromXYA(-107.309311,242.988007,-2.673822)),
   (254, Frame2D.fromXYA(-109.084908,241.640610,-2.648838)),
   (255, Frame2D.fromXYA(-110.816521,240.231049,-2.626271)),
   (256, Frame2D.fromXYA(-112.602486,238.809052,-2.606146)),
   (257, Frame2D.fromXYA(-113.564148,238.098663,-2.596233)),
   (258, Frame2D.fromXYA(-115.317383,236.641327,-2.575053)),
   (259, Frame2D.fromXYA(-116.882309,235.269958,-2.552911)),
   (260, Frame2D.fromXYA(-117.259132,234.927933,-2.547310)),
   (261, Frame2D.fromXYA(-119.614624,232.928558,-2.523707)),
   (262, Frame2D.fromXYA(-121.644028,231.242615,-2.509762)),
   (263, Frame2D.fromXYA(-122.622185,230.345520,-2.500434)),
   (264, Frame2D.fromXYA(-124.485802,228.548523,-2.481799)),
   (265, Frame2D.fromXYA(-126.503448,226.453751,-2.455857)),
   (266, Frame2D.fromXYA(-126.832146,226.072693,-2.449433)),
   (267, Frame2D.fromXYA(-128.381851,224.419647,-2.430033)),
   (268, Frame2D.fromXYA(-130.025955,222.672989,-2.411892)),
   (269, Frame2D.fromXYA(-130.761795,221.803070,-2.401549)),
   (270, Frame2D.fromXYA(-132.223145,220.147629,-2.382943)),
   (271, Frame2D.fromXYA(-133.762344,218.412979,-2.368479)),
   (272, Frame2D.fromXYA(-134.602142,217.482544,-2.362034)),
   (273, Frame2D.fromXYA(-136.095413,215.781876,-2.351030)),
   (274, Frame2D.fromXYA(-137.621063,214.029663,-2.339654)),
   (275, Frame2D.fromXYA(-139.173767,212.221054,-2.328179)),
   (276, Frame2D.fromXYA(-140.255539,210.913651,-2.318757)),
   (277, Frame2D.fromXYA(-141.346664,209.522903,-2.309383)),
   (278, Frame2D.fromXYA(-142.900986,207.300873,-2.287736)),
   (279, Frame2D.fromXYA(-143.329651,206.522339,-2.275471)),
   (280, Frame2D.fromXYA(-144.209259,204.803757,-2.250117)),
   (281, Frame2D.fromXYA(-144.464859,204.363052,-2.244668)),
   (282, Frame2D.fromXYA(-145.403610,202.550491,-2.221967)),
   (283, Frame2D.fromXYA(-146.321655,200.732925,-2.199012)),
   (284, Frame2D.fromXYA(-147.412704,198.439926,-2.170426)),
   (285, Frame2D.fromXYA(-147.949326,197.439621,-2.161496)),
   (286, Frame2D.fromXYA(-148.856598,195.789597,-2.147668)),
   (287, Frame2D.fromXYA(-150.171600,193.199707,-2.123360)),
   (288, Frame2D.fromXYA(-150.436920,192.637405,-2.118172)),
   (289, Frame2D.fromXYA(-151.518951,189.997406,-2.091985)),
   (290, Frame2D.fromXYA(-152.365204,187.734467,-2.068585)),
   (291, Frame2D.fromXYA(-152.811905,186.623581,-2.057370)),
   (292, Frame2D.fromXYA(-153.770721,184.224594,-2.039180)),
   (293, Frame2D.fromXYA(-154.242432,182.993362,-2.029647)),
   (294, Frame2D.fromXYA(-154.869797,181.203018,-2.016379)),
   (295, Frame2D.fromXYA(-155.935303,178.173416,-1.994143)),
   (296, Frame2D.fromXYA(-156.687531,175.836929,-1.977347)),
   (297, Frame2D.fromXYA(-157.006531,174.679321,-1.967700)),
   (298, Frame2D.fromXYA(-157.579361,172.408234,-1.947922)),
   (299, Frame2D.fromXYA(-157.988968,170.824158,-1.933176)),
   (300, Frame2D.fromXYA(-158.772583,167.966354,-1.913703)),
   (301, Frame2D.fromXYA(-159.064453,166.880035,-1.906280)),
   (302, Frame2D.fromXYA(-159.499344,164.643677,-1.884509)),
   (303, Frame2D.fromXYA(-160.018875,162.439163,-1.872204)),
   (304, Frame2D.fromXYA(-160.312561,161.284363,-1.867553)),
   (305, Frame2D.fromXYA(-160.444717,160.734406,-1.864972)),
   (306, Frame2D.fromXYA(-161.449295,156.457397,-1.844979)),
   (307, Frame2D.fromXYA(-161.762985,155.174652,-1.841214)),
   (308, Frame2D.fromXYA(-162.363907,152.807892,-1.835876)),
   (309, Frame2D.fromXYA(-162.749283,150.708115,-1.822495)),
   (310, Frame2D.fromXYA(-162.770020,150.186081,-1.817050)),
   (311, Frame2D.fromXYA(-162.413147,148.303711,-1.776179)),
   (312, Frame2D.fromXYA(-162.001877,146.695862,-1.740860)),
   (313, Frame2D.fromXYA(-161.530624,145.041534,-1.704943)),
   (314, Frame2D.fromXYA(-161.294067,144.127365,-1.686132)),
   (315, Frame2D.fromXYA(-161.297607,142.301132,-1.670645)),
   (316, Frame2D.fromXYA(-161.532059,138.736740,-1.659653)),
   (317, Frame2D.fromXYA(-161.565628,137.483902,-1.654080)),
   (318, Frame2D.fromXYA(-161.566177,134.966522,-1.639612)),
   (319, Frame2D.fromXYA(-161.355804,132.753571,-1.618183)),
   (320, Frame2D.fromXYA(-161.156631,131.622940,-1.603788)),
   (321, Frame2D.fromXYA(-160.907059,129.168121,-1.581544)),
   (322, Frame2D.fromXYA(-160.804901,126.726120,-1.569437)),
   (323, Frame2D.fromXYA(-160.745514,125.406464,-1.562980)),
   (324, Frame2D.fromXYA(-160.606613,122.645905,-1.554504)),
   (325, Frame2D.fromXYA(-160.479126,119.954689,-1.546663)),
   (326, Frame2D.fromXYA(-160.228165,117.581345,-1.531398)),
   (327, Frame2D.fromXYA(-160.046799,116.458939,-1.522509)),
   (328, Frame2D.fromXYA(-159.587067,114.162666,-1.502100)),
   (329, Frame2D.fromXYA(-159.108368,112.004410,-1.480440)),
   (330, Frame2D.fromXYA(-158.749512,110.337730,-1.465929)),
   (331, Frame2D.fromXYA(-158.361740,108.554031,-1.452968)),
   (332, Frame2D.fromXYA(-157.882767,106.076759,-1.438879)),
   (333, Frame2D.fromXYA(-157.624313,104.980118,-1.431667)),
   (334, Frame2D.fromXYA(-156.929565,101.790070,-1.413280)),
   (335, Frame2D.fromXYA(-156.393646,99.086975,-1.402069)),
   (336, Frame2D.fromXYA(-156.248550,98.407440,-1.399928)),
   (337, Frame2D.fromXYA(-155.558426,95.221306,-1.390308)),
   (338, Frame2D.fromXYA(-155.113174,93.581802,-1.380002)),
   (339, Frame2D.fromXYA(-154.243576,91.269096,-1.352675)),
   (340, Frame2D.fromXYA(-154.043869,90.727470,-1.347998)),
   (341, Frame2D.fromXYA(-153.358231,88.639084,-1.334553)),
   (342, Frame2D.fromXYA(-152.390991,86.105774,-1.310367)),
   (343, Frame2D.fromXYA(-152.187408,85.639099,-1.304744)),
   (344, Frame2D.fromXYA(-151.318390,83.665230,-1.283954)),
   (345, Frame2D.fromXYA(-150.131317,81.679001,-1.250263)),
   (346, Frame2D.fromXYA(-149.655899,80.841110,-1.238191)),
   (347, Frame2D.fromXYA(-148.701920,79.059502,-1.216799)),
   (348, Frame2D.fromXYA(-147.895081,77.349579,-1.206001)),
   (349, Frame2D.fromXYA(-147.369736,76.211182,-1.199081)),
   (350, Frame2D.fromXYA(-146.009933,73.509789,-1.177948)),
   (351, Frame2D.fromXYA(-144.962753,71.442757,-1.164662)),
   (352, Frame2D.fromXYA(-144.374786,70.467316,-1.153454)),
   (353, Frame2D.fromXYA(-143.124359,68.833969,-1.125935)),
   (354, Frame2D.fromXYA(-142.177856,67.479149,-1.109296)),
   (355, Frame2D.fromXYA(-141.549393,66.531273,-1.100397)),
   (356, Frame2D.fromXYA(-140.044708,64.348244,-1.075118)),
   (357, Frame2D.fromXYA(-138.567535,62.375950,-1.052693)),
   (358, Frame2D.fromXYA(-137.023987,60.228905,-1.034787)),
   (359, Frame2D.fromXYA(-136.226028,59.168636,-1.024121)),
   (360, Frame2D.fromXYA(-135.100525,57.586002,-1.011561)),
   (361, Frame2D.fromXYA(-133.137665,54.965073,-0.994073)),
   (362, Frame2D.fromXYA(-132.363968,54.049423,-0.984627)),
   (363, Frame2D.fromXYA(-130.835358,52.135941,-0.968041)),
   (364, Frame2D.fromXYA(-129.959732,51.146511,-0.958838)),
   (365, Frame2D.fromXYA(-128.343353,49.291679,-0.939087)),
   (366, Frame2D.fromXYA(-126.640884,47.201412,-0.926778)),
   (367, Frame2D.fromXYA(-124.895523,44.991734,-0.919518)),
   (368, Frame2D.fromXYA(-123.951691,43.811993,-0.915954)),
   (369, Frame2D.fromXYA(-122.392242,41.921997,-0.909489)),
   (370, Frame2D.fromXYA(-120.817436,40.054329,-0.901664)),
   (371, Frame2D.fromXYA(-120.404472,39.583447,-0.899353)),
   (372, Frame2D.fromXYA(-118.345215,37.304241,-0.884201)),
   (373, Frame2D.fromXYA(-116.656525,35.522942,-0.872093)),
   (374, Frame2D.fromXYA(-115.908768,34.846416,-0.863995)),
   (375, Frame2D.fromXYA(-114.473473,34.015163,-0.836222)),
   (376, Frame2D.fromXYA(-113.805069,33.686249,-0.821463)),
   (377, Frame2D.fromXYA(-111.721970,32.831341,-0.772822)),
   (378, Frame2D.fromXYA(-111.018539,32.572693,-0.756441)),
   (379, Frame2D.fromXYA(-108.962311,31.670881,-0.714224)),
   (380, Frame2D.fromXYA(-106.499481,29.631660,-0.707925)),
   (381, Frame2D.fromXYA(-105.861450,29.121899,-0.705857)),
   (382, Frame2D.fromXYA(-103.496696,27.168308,-0.701175)),
   (383, Frame2D.fromXYA(-101.199722,25.312332,-0.695759)),
   (384, Frame2D.fromXYA(-100.121986,24.460546,-0.691619)),
   (385, Frame2D.fromXYA(-98.055870,22.925524,-0.680950)),
   (386, Frame2D.fromXYA(-96.010628,21.574314,-0.663984)),
   (387, Frame2D.fromXYA(-95.037933,20.985960,-0.654896)),
   (388, Frame2D.fromXYA(-92.964882,19.769196,-0.635598)),
   (389, Frame2D.fromXYA(-90.430237,18.354958,-0.609852)),
   (390, Frame2D.fromXYA(-88.930115,17.669987,-0.590380)),
   (391, Frame2D.fromXYA(-87.882378,17.207024,-0.579065)),
   (392, Frame2D.fromXYA(-85.224564,15.897927,-0.555194)),
   (393, Frame2D.fromXYA(-83.595200,15.081894,-0.542469)),
   (394, Frame2D.fromXYA(-82.511154,14.575912,-0.534487)),
   (395, Frame2D.fromXYA(-79.651039,13.257219,-0.513581)),
   (396, Frame2D.fromXYA(-78.016052,12.586823,-0.500253)),
   (397, Frame2D.fromXYA(-75.632088,11.873997,-0.469913)),
   (398, Frame2D.fromXYA(-74.668953,11.548325,-0.457882)),
   (399, Frame2D.fromXYA(-73.019669,10.928215,-0.444487)),
   (400, Frame2D.fromXYA(-70.206825,9.847302,-0.428992)),
   (401, Frame2D.fromXYA(-69.024620,9.406363,-0.421638)),
   (402, Frame2D.fromXYA(-66.695076,8.486573,-0.411744)),
   (403, Frame2D.fromXYA(-65.623848,8.099864,-0.406387)),
   (404, Frame2D.fromXYA(-64.077240,7.584935,-0.394825)),
   (405, Frame2D.fromXYA(-61.319870,6.710036,-0.376678)),
   (406, Frame2D.fromXYA(-59.096275,6.029270,-0.363208)),
   (407, Frame2D.fromXYA(-58.012573,5.724632,-0.354830)),
   (408, Frame2D.fromXYA(-55.846226,5.233343,-0.335473)),
   (409, Frame2D.fromXYA(-54.133022,4.819725,-0.323028)),
   (410, Frame2D.fromXYA(-53.080322,4.625555,-0.313319)),
   (411, Frame2D.fromXYA(-50.608452,4.330933,-0.285404)),
   (412, Frame2D.fromXYA(-48.437580,3.959405,-0.268370)),
   (413, Frame2D.fromXYA(-47.431595,3.851949,-0.256619)),
   (414, Frame2D.fromXYA(-45.320877,3.740285,-0.232380)),
   (415, Frame2D.fromXYA(-43.679630,3.629228,-0.214688)),
   (416, Frame2D.fromXYA(-41.744972,3.374311,-0.202430)),
   (417, Frame2D.fromXYA(-39.315834,3.073809,-0.188811)),
   (418, Frame2D.fromXYA(-36.942764,2.818426,-0.173519)),
   (419, Frame2D.fromXYA(-34.607918,2.675833,-0.154154)),
   (420, Frame2D.fromXYA(-33.597786,2.673509,-0.144254)),
   (421, Frame2D.fromXYA(-31.334698,2.613282,-0.125966)),
   (422, Frame2D.fromXYA(-28.875925,2.551078,-0.108782)),
   (423, Frame2D.fromXYA(-27.619326,2.537589,-0.098987)),
   (424, Frame2D.fromXYA(-25.156336,2.585459,-0.079369)),
   (425, Frame2D.fromXYA(-22.703333,2.663113,-0.062602)),
   (426, Frame2D.fromXYA(-22.013475,2.693192,-0.058258)),
   (427, Frame2D.fromXYA(-18.931061,2.777831,-0.038613)),
   (428, Frame2D.fromXYA(-16.544037,2.962731,-0.021343)),
   (429, Frame2D.fromXYA(-15.283535,3.050961,-0.014364)),
   (430, Frame2D.fromXYA(-12.650080,3.282099,0.003294)),
   (431, Frame2D.fromXYA(-10.583817,3.436890,0.014206)),
   (432, Frame2D.fromXYA(-8.771101,3.601903,0.025197)),
   (433, Frame2D.fromXYA(-6.275209,3.840204,0.039648)),
   (434, Frame2D.fromXYA(-3.841152,4.204288,0.057532)),
   (435, Frame2D.fromXYA(-2.597176,4.359362,0.064087)),
   (436, Frame2D.fromXYA(0.093679,4.599052,0.071254)),
   (437, Frame2D.fromXYA(2.710876,4.895014,0.079592)),
   (438, Frame2D.fromXYA(4.932744,5.321552,0.094813)),
   (439, Frame2D.fromXYA(5.720722,5.714037,0.110349)),
   (440, Frame2D.fromXYA(7.014661,6.431636,0.140469)),
   (441, Frame2D.fromXYA(8.290217,7.209489,0.171602)),
   (442, Frame2D.fromXYA(9.198492,7.793417,0.194635)),
   (443, Frame2D.fromXYA(10.268259,8.490560,0.220347)),
   (444, Frame2D.fromXYA(12.862708,9.226366,0.230957)),
   (445, Frame2D.fromXYA(14.323256,9.581444,0.232244)),
   (446, Frame2D.fromXYA(16.925674,10.309944,0.241526)),
   (447, Frame2D.fromXYA(18.189697,10.665023,0.244784)),
   (448, Frame2D.fromXYA(20.376797,11.470398,0.261567)),
   (449, Frame2D.fromXYA(22.344488,12.337081,0.284227)),
   (450, Frame2D.fromXYA(25.095102,13.440198,0.304336)),
   (451, Frame2D.fromXYA(27.532713,14.457230,0.319922)),
   (452, Frame2D.fromXYA(28.174828,14.702248,0.324262)),
   (453, Frame2D.fromXYA(30.712584,15.814907,0.338502)),
   (454, Frame2D.fromXYA(31.783989,16.330961,0.348048)),
   (455, Frame2D.fromXYA(33.782806,17.453636,0.370114)),
   (456, Frame2D.fromXYA(35.671341,18.961285,0.406641)),
   (457, Frame2D.fromXYA(37.380985,20.185724,0.434918)),
   (458, Frame2D.fromXYA(38.386993,20.860617,0.445907)),
   (459, Frame2D.fromXYA(39.459435,21.517258,0.454147)),
   (460, Frame2D.fromXYA(42.326752,23.203854,0.472171)),
   (461, Frame2D.fromXYA(43.916153,24.232639,0.484243)),
   (462, Frame2D.fromXYA(46.288788,25.652582,0.495336)),
   (463, Frame2D.fromXYA(48.592110,27.053619,0.506330)),
   (464, Frame2D.fromXYA(49.668358,27.692741,0.509994)),
   (465, Frame2D.fromXYA(51.210552,28.654608,0.517306)),
   (466, Frame2D.fromXYA(53.839771,30.371790,0.531513)),
   (467, Frame2D.fromXYA(54.940872,31.102619,0.537261)),
   (468, Frame2D.fromXYA(56.764923,32.435406,0.552729)),
   (469, Frame2D.fromXYA(58.305309,33.754391,0.572599)),
   (470, Frame2D.fromXYA(58.809063,34.127392,0.575541)),
   (471, Frame2D.fromXYA(59.652130,34.789909,0.582698)),
   (472, Frame2D.fromXYA(62.376064,37.240311,0.613621)),
   (473, Frame2D.fromXYA(64.013870,38.622719,0.625422)),
   (474, Frame2D.fromXYA(64.769516,39.386391,0.635727)),
   (475, Frame2D.fromXYA(66.360374,40.939732,0.654681)),
   (476, Frame2D.fromXYA(67.463089,42.056805,0.669837)),
   (477, Frame2D.fromXYA(68.807938,43.282455,0.679429)),
   (478, Frame2D.fromXYA(70.609116,44.948063,0.693923)),
   (479, Frame2D.fromXYA(72.266937,46.579628,0.710641)),
   (480, Frame2D.fromXYA(73.041267,47.410007,0.718589)),
   (481, Frame2D.fromXYA(74.070267,48.698093,0.735460)),
   (482, Frame2D.fromXYA(75.788620,50.816093,0.764801)),
   (483, Frame2D.fromXYA(76.519974,51.684048,0.773534)),
   (484, Frame2D.fromXYA(77.839615,53.461346,0.795329)),
   (485, Frame2D.fromXYA(79.022232,55.369907,0.821718)),
   (486, Frame2D.fromXYA(80.395264,57.256905,0.841068)),
   (487, Frame2D.fromXYA(81.061043,58.175545,0.849475)),
   (488, Frame2D.fromXYA(82.233055,59.885365,0.865828)),
   (489, Frame2D.fromXYA(83.462524,61.889153,0.888255)),
   (490, Frame2D.fromXYA(84.141090,62.949387,0.896690)),
   (491, Frame2D.fromXYA(85.494102,65.068748,0.913493)),
   (492, Frame2D.fromXYA(86.810051,67.144867,0.929682)),
   (493, Frame2D.fromXYA(87.446411,68.239838,0.938137)),
   (494, Frame2D.fromXYA(88.654808,70.227608,0.955042)),
   (495, Frame2D.fromXYA(89.984207,72.351562,0.966640)),
   (496, Frame2D.fromXYA(91.352272,74.677948,0.979962)),
   (497, Frame2D.fromXYA(92.074921,75.855827,0.985973)),
   (498, Frame2D.fromXYA(93.390060,78.140564,0.997972)),
   (499, Frame2D.fromXYA(94.535355,80.377426,1.015457)),
   (500, Frame2D.fromXYA(95.440773,82.113243,1.026148)),
   (501, Frame2D.fromXYA(96.611588,84.328491,1.039228)),
   (502, Frame2D.fromXYA(97.427803,85.963058,1.048999)),
   (503, Frame2D.fromXYA(98.183586,87.551888,1.057749)),
   (504, Frame2D.fromXYA(98.785439,89.067451,1.073232)),
   (505, Frame2D.fromXYA(99.801277,91.654800,1.096211)),
   (506, Frame2D.fromXYA(100.052315,92.229210,1.098967)),
   (507, Frame2D.fromXYA(101.125717,94.567474,1.106820)),
   (508, Frame2D.fromXYA(102.451965,97.409775,1.115308)),
   (509, Frame2D.fromXYA(102.672249,97.930573,1.117516)),
   (510, Frame2D.fromXYA(103.687485,100.219177,1.125303)),
   (511, Frame2D.fromXYA(104.486748,102.338707,1.138766)),
   (512, Frame2D.fromXYA(105.229019,104.489792,1.157794)),
   (513, Frame2D.fromXYA(105.574974,105.562042,1.165908)),
   (514, Frame2D.fromXYA(106.185341,107.754372,1.184852)),
   (515, Frame2D.fromXYA(106.816109,110.064667,1.203589)),
   (516, Frame2D.fromXYA(107.205399,111.654785,1.216166)),
   (517, Frame2D.fromXYA(107.629425,113.772079,1.236372)),
   (518, Frame2D.fromXYA(107.860039,115.546059,1.254801)),
   (519, Frame2D.fromXYA(108.288940,117.387405,1.267280)),
   (520, Frame2D.fromXYA(108.792007,119.323914,1.277453)),
   (521, Frame2D.fromXYA(109.442131,122.400902,1.294686)),
   (522, Frame2D.fromXYA(109.980751,124.919670,1.307508)),
   (523, Frame2D.fromXYA(110.154282,126.104980,1.316621)),
   (524, Frame2D.fromXYA(110.254227,128.272125,1.341633)),
   (525, Frame2D.fromXYA(110.390511,130.110306,1.358869)),
   (526, Frame2D.fromXYA(110.506439,131.364197,1.369773)),
   (527, Frame2D.fromXYA(110.845604,134.430313,1.390033)),
   (528, Frame2D.fromXYA(111.071869,136.930908,1.406558)),
   (529, Frame2D.fromXYA(111.166702,138.190872,1.413648)),
   (530, Frame2D.fromXYA(111.220894,139.317215,1.422489)),
   (531, Frame2D.fromXYA(111.368492,142.208191,1.443581)),
   (532, Frame2D.fromXYA(111.434174,144.029572,1.455020)),
   (533, Frame2D.fromXYA(111.413712,146.556168,1.473210)),
   (534, Frame2D.fromXYA(111.469513,148.818817,1.488153)),
   (535, Frame2D.fromXYA(111.471985,150.073807,1.494777)),
   (536, Frame2D.fromXYA(111.384590,152.091034,1.510759)),
   (537, Frame2D.fromXYA(111.122116,153.982422,1.534335)),
   (538, Frame2D.fromXYA(110.925049,154.805222,1.546497)),
   (539, Frame2D.fromXYA(110.710564,156.815353,1.565982)),
   (540, Frame2D.fromXYA(110.411705,158.820389,1.585556)),
   (541, Frame2D.fromXYA(110.054054,160.879517,1.605987)),
   (542, Frame2D.fromXYA(109.947029,161.448135,1.611116)),
   (543, Frame2D.fromXYA(109.583481,164.310074,1.628325)),
   (544, Frame2D.fromXYA(109.249115,166.551239,1.643557)),
   (545, Frame2D.fromXYA(109.059685,167.661514,1.651395)),
   (546, Frame2D.fromXYA(108.616402,169.697083,1.671231)),
   (547, Frame2D.fromXYA(108.241837,171.298660,1.687639)),
   (548, Frame2D.fromXYA(107.767471,172.884399,1.705243)),
   (549, Frame2D.fromXYA(107.180992,175.274292,1.724482)),
   (550, Frame2D.fromXYA(106.561562,177.708618,1.740190)),
   (551, Frame2D.fromXYA(106.253883,178.925034,1.746965)),
   (552, Frame2D.fromXYA(105.588531,181.486237,1.761433)),
   (553, Frame2D.fromXYA(104.896996,183.899612,1.778417)),
   (554, Frame2D.fromXYA(104.540771,184.915482,1.786528)),
   (555, Frame2D.fromXYA(103.732361,187.172318,1.805524)),
   (556, Frame2D.fromXYA(102.808403,189.527863,1.828130)),
   (557, Frame2D.fromXYA(101.921577,191.887878,1.844962)),
   (558, Frame2D.fromXYA(101.252632,193.712067,1.856415)),
   (559, Frame2D.fromXYA(100.640671,195.432190,1.867663)),
   (560, Frame2D.fromXYA(100.246742,196.548218,1.872955)),
   (561, Frame2D.fromXYA(99.348709,198.760040,1.887093)),
   (562, Frame2D.fromXYA(98.354446,201.211029,1.900308)),
   (563, Frame2D.fromXYA(97.046638,204.406647,1.914442)),
   (564, Frame2D.fromXYA(96.805984,205.049789,1.917153)),
   (565, Frame2D.fromXYA(95.832634,207.496338,1.923871)),
   (566, Frame2D.fromXYA(94.604301,209.975449,1.945467)),
   (567, Frame2D.fromXYA(94.032204,210.896667,1.958201)),
   (568, Frame2D.fromXYA(93.004608,212.840637,1.973793)),
   (569, Frame2D.fromXYA(92.157631,214.386124,1.985880)),
   (570, Frame2D.fromXYA(91.613754,215.388153,1.993465)),
   (571, Frame2D.fromXYA(90.094215,217.996674,2.016737)),
   (572, Frame2D.fromXYA(88.898232,220.061752,2.032219)),
   (573, Frame2D.fromXYA(88.375221,220.985962,2.038954)),
   (574, Frame2D.fromXYA(87.143097,222.960205,2.053233)),
   (575, Frame2D.fromXYA(86.268089,224.275208,2.064407)),
   (576, Frame2D.fromXYA(85.655640,225.227524,2.072566)),
   (577, Frame2D.fromXYA(84.015762,227.754395,2.089055)),
   (578, Frame2D.fromXYA(82.599274,229.834763,2.103579)),
   (579, Frame2D.fromXYA(81.207008,231.552429,2.122570)),
   (580, Frame2D.fromXYA(80.450310,232.411148,2.134144)),
   (581, Frame2D.fromXYA(79.353386,233.719345,2.150284)),
   (582, Frame2D.fromXYA(77.590324,235.474167,2.179081)),
   (583, Frame2D.fromXYA(76.849625,236.259079,2.190683)),
   (584, Frame2D.fromXYA(75.304726,238.078659,2.207650)),
   (585, Frame2D.fromXYA(73.619591,239.797729,2.229068)),
   (586, Frame2D.fromXYA(73.182846,240.241089,2.233675)),
   (587, Frame2D.fromXYA(71.083511,242.421753,2.255188)),
   (588, Frame2D.fromXYA(69.344009,244.070633,2.275325)),
   (589, Frame2D.fromXYA(68.509140,244.842056,2.285095)),
   (590, Frame2D.fromXYA(66.653847,246.715851,2.297938)),
   (591, Frame2D.fromXYA(64.872597,248.489471,2.311524)),
   (592, Frame2D.fromXYA(64.418617,248.921951,2.314648)),
   (593, Frame2D.fromXYA(62.029610,251.231705,2.328905)),
   (594, Frame2D.fromXYA(60.125214,252.967773,2.347075)),
   (595, Frame2D.fromXYA(58.180111,254.756119,2.358762)),
   (596, Frame2D.fromXYA(57.171623,255.602371,2.366529)),
   (597, Frame2D.fromXYA(55.807442,256.616821,2.382616)),
   (598, Frame2D.fromXYA(53.556526,258.160126,2.408393)),
   (599, Frame2D.fromXYA(52.594975,258.778290,2.419101))]
