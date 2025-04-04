# Embodied-AI-Paper-Research
A paper research of Embodied AI.

# 具身感知 (Embodied Sensing):

## 一，Object Sensing:

### 1, Geometric shapes (including point clouds, grids, voxels, depth maps, whose downstream tasks are object pose estimation and grasping)：

#### (1), https://adioshun.gitbooks.io/deep_drive/content/intro3d-cloudpoint.html
#### (2), UMPNet: Universal Manipulation Policy Network for Articulated Objects. 2022 RA-L https://arxiv.org/pdf/2109.05668
#### (3), Tactile-RL for Insertion: Generalization to Objects of Unknown Geometry https://arxiv.org/pdf/2104.01167
#### (4), Pointnet: Deep learning on point sets for 3d classification and segmentation. 2017 CVPR https://arxiv.org/pdf/1612.00593
#### (5), Pointnet++: Deep hierarchical feature learning on point sets in a metric space. 2017 NIPS https://arxiv.org/pdf/1706.02413
#### (6), Meshnet: Mesh neural network for 3d shape representation. 2019 AAAI  https://export.arxiv.org/pdf/1811.11424
#### (7), VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. 2018 CVPR  https://arxiv.org/pdf/1711.06396
#### (8), DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. 2019 CVPR https://arxiv.org/pdf/1901.05103
#### (9), Occupancy Networks: Learning 3D Reconstruction in Function Space. 2019 CVPR https://arxiv.org/pdf/1812.03828
#### (10), Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation. 2019 CVPR https://arxiv.org/pdf/1901.02970

### 2, Articulated structure (including two technical routes of directly modeling joint parameters and modeling displacement changes, with downstream tasks of interactive perception and object availability prediction):

#### (1), CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models. 2024 ICRA https://arxiv.org/pdf/2403.08248
#### (2), Toward Real-World Category-Level Articulation Pose Estimation. 2022 TIP https://arxiv.org/pdf/2105.03260
#### (3), AKB-48: A Real-World Articulated Object Knowledge Base. 2022 CVPR https://arxiv.org/pdf/2202.08432
#### (4), CAGE: Controllable Articulation GEneration. 2024 CVPR https://arxiv.org/pdf/2312.09570
#### (5), Category-Level Articulated Object Pose Estimation. 2020 CVPR https://arxiv.org/pdf/1912.11913
#### (6), Self-supervised Neural Articulated Shape and Appearance Models. 2022 CVPR https://arxiv.org/pdf/2205.08525
#### (7), SAGCI-System: Towards Sample-Efficient, Generalizable, Compositional and Incremental Robot Learning. 2022 ICRA https://arxiv.org/pdf/2111.14693
#### (8), Where2Act: From Pixels to Actions for Articulated 3D Objects. 2024 ICCV https://arxiv.org/pdf/2101.02692

### 3, Physical properties (including touch, torque, temperature, material, hardness, etc.)

#### (1), Imagebind: One embedding space to bind them all. 2023 CVPR https://arxiv.org/pdf/2305.05665
#### (2), Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. 2024 ICLR https://arxiv.org/pdf/2310.01852
#### (3), Tactile-rl for insertion: Generalization to objects of unknown geometry. 2024 ICRA https://arxiv.org/pdf/2104.01167
#### (4), Precise Robotic Needle-Threading with Tactile Perception and Reinforcement Learning. 2023 CoRL https://arxiv.org/pdf/2311.02396

## 二, Scene Sensing:

### 1, Scene Reconstruction(The key technology is SLAM)：

#### (1), Learning the next best view for 3d point clouds via topological features. 2021 ICRA https://ar5iv.labs.arxiv.org/html/2103.02789
#### (2), Bag of views: An appearance-based approach to next-best-view planning for 3d reconstruction. 2023 RAL https://arxiv.org/pdf/2307.05832
#### (3), Object-aware guidance for autonomous scene reconstruction. 2018 TOG https://arxiv.org/pdf/1805.07794
#### (4), Multi-robot collaborative dense scene reconstruction. 2019 TOG https://dl.acm.org/doi/pdf/10.1145/3306346.3322942
#### (5), Active neural localization. 2018 arXiv https://arxiv.org/pdf/1801.08214

### 2, Scene Understanding(Including object recognition, spatial relationship inference, scene change detection, and scene dynamic perception)

#### (1), Learning instance segmentation by interaction. 2018 CVPR https://arxiv.org/pdf/1806.08354
#### (2), You only look once: Unified, real-time object detection. 2016 CVPR https://export.arxiv.org/pdf/1506.02640v3
#### (3), Mask r-cnn. 2017 ICCV  https://ieeexplore.ieee.org/document/8237584
#### (4), Deep residual learning for image recognition. 2016 CVPR https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780459
#### (5), Move to see better: Self-improving embodied object detection. 2020 arXiv https://arxiv.org/pdf/2012.00057
#### (6), Rel3d: A minimally contrastive benchmark for grounding spatial relations in 3d. 2020 NIPS https://arxiv.org/pdf/2012.01634
#### (7), Spatialsense: An adversarially crowdsourced benchmark for spatial relation recognition. 2019 ICCV https://arxiv.org/pdf/1908.02660
#### (8), The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. https://arxiv.org/pdf/1811.00982
#### (9), The robotic vision scene understanding challenge. 2020 arXiv https://arxiv.org/pdf/2009.05246
#### (10), Changesim: Towards end-to-end online scene change detection in industrial indoor environments. 2021 IROS https://ieeexplore.ieee.org/abstract/document/9636350
#### (11), Cdnet++: Improved change detection with deep neural network feature correlation. 2020 IJCNN https://ieeexplore.ieee.org/abstract/document/9207306
#### (12), Weakly supervised silhouette-based semantic scene change detection. 2020 ICRA https://arxiv.org/pdf/1811.11985
#### (13), Continuous scene representations for embodied ai. 2022 CVPR https://arxiv.org/pdf/2203.17251
#### (14), Object-level change detection with a dual correlation attention-guided detector. 2021 ISPRS https://www.sciencedirect.com/science/article/pii/S0924271621001271
#### (15), 4d panoptic scene graph generation. 2024 NIPS https://arxiv.org/pdf/2405.10305

## 三, Behavior Sensing:

### 1, Gesture Detection

### 2, Human Pose Detection

#### (1), ProxEmo: Gait-based Emotion Learning and Multi-view Proxemic Fusion for Socially-Aware Robot Navigation. 2020 IROS https://arxiv.org/pdf/2003.01062
#### (2), Recurrent neural network for motion trajectory prediction in human-robot collaborative assembly. 2020 CIRP. https://www.sciencedirect.com/science/article/pii/S0007850620300998

### 3, Human Behavior Understanding

#### (1), Motiongpt: Human motion as a foreign language. 2024 NIPS https://arxiv.org/pdf/2306.14795
#### (2), MotionLLM: Understanding Human Behaviors from Human Motions and Videos. 2024 arXiv https://arxiv.org/pdf/2405.20340

## 四, Representation Sensing:

### 1, Emotion Computing

#### (1), Facial expression recognition with visual transformers and attentional selective fusion. 2021 IEEE Transactions on Affective Computing https://arxiv.org/pdf/2103.16854
#### (2), Region attention networks for pose and occlusion robust facial expression recognition. 2020 IEEE Transactions on Image Processing https://arxiv.org/pdf/1905.04075
#### (3), Edge-AI-driven framework with efficient mobile network design for facial expression recognition. 2023 ACM Transactions on Embedded Computing Systems https://dl.acm.org/doi/10.1145/3587038

### 2, Intention Detection

### 3, Referential expression

#### (1), Mattnet: Modular attention network for referring expression comprehension. 2018 Proceedings of the IEEE conference on computer vision and pattern recognition https://arxiv.org/pdf/1801.08186

#### (2), Dynamic graph attention for referring expression comprehension. 2019 Proceedings of the IEEE/CVF International Conference on Computer Vision https://arxiv.org/pdf/1909.08164


# 具身推理 (Embodied Reasoning):

##  一, Task Planning

###  1, Expert system

####  (1), STRIPS: A New Approach to the Application of .Theorem Proving to Problem Solving. 1971 IJCAI. https://www.sciencedirect.com/science/article/pdf/pii/0004370271900105
####  (2), PRODIGY: an integrated architecture for planning and learning. 1991 SIGART Bull. https://dl.acm.org/doi/pdf/10.1145/122344.122353
####  (3), SHOP2: An HTN Planning System. 2003 arXiv. https://arxiv.org/pdf/1106.4869

###  2, Unified Modeling Language

#### (1), PDDL-the planning domain definition language. 1998 ICAPS https://www.academia.edu/23178990/PDDL_The_Planning_Domain_Definition_Language#:~:text=This%20manual%20describes%20the%20syntax%20of%20PDDL%2C%20the,the%20problem-specification%20language%20for%20the%20AIPS-98%20planning%20competition.

#### (2), What Is Answer Set Programming?. 2008. AAAI https://www.cs.utexas.edu/~vl/papers/wiasp.pdf

###  3, DL-based Task Planning
#### (1), Regression Planning Networks. 2019. NeurIPS https://arxiv.org/pdf/1909.13072

###  4, LLM as Translator
#### (1), LLM+P: Empowering Large Language Models with Optimal Planning Proficiency. 2023 arXiv.   https://arxiv.org/pdf/2304.11477

###  5, LLM as Planner

#### (1), Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents. 2022. arXiv. https://arxiv.org/pdf/2201.07207
#### (2), ProgPrompt: Generating Situated Robot Task Plans using Large Language Models. 2023 Autonomous Robots.  https://arxiv.org/pdf/2209.11302
#### (3), PaLM-E: An Embodied Multimodal Language Model. 2023. arXiv.  https://arxiv.org/pdf/2303.03378
#### (4), Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. 2022. arXiv.  https://arxiv.org/pdf/2204.01691
#### (5), EgoPlan-Bench: Benchmarking Multimodal Large Language Models for Human-Level Planning. 2023. arXiv.  https://arxiv.org/pdf/2312.06722
#### (6), BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments. 2021. arXiv.  https://arxiv.org/pdf/2108.03332
#### (7), Watch-and-help: A challenge for social perception and human-ai collaboration. 2020. arXiv.  https://arxiv.org/pdf/2010.09890
#### (8), ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. 2020. CVPR2020.  https://arxiv.org/pdf/1912.01734

##  二, Embodied Navigation

###  1, Rule-based Navigation
###  2, Learning-based Navigation

#### (1), Cognitive mapping and planning for visual navigation. 2020 IJCV https://arxiv.org/pdf/1702.03920
#### (2), A behavioral approach to visual navigation with graph localizationnetworks. 2019 RSS https://arxiv.org/pdf/1903.00445
#### (3), Occupancy anticipation for efficient exploration and navigation. 2020 ECCV  https://arxiv.org/pdf/2008.09285
#### (4), Visual semantic navigation using scene priors. 2018 arxiv https://arxiv.org/pdf/1810.06543
#### (5), Bayesian relational memory for semantic visual navigation. 2019 ICCV https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9009539
#### (6), Learning to act by predicting the future. 2017 ICLR https://arxiv.org/pdf/1611.01779
#### (7), Benchmarking classic and learned navigation in complex 3d environments. 2019 arxiv https://arxiv.org/pdf/1901.10915
#### (8), DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion  Frames. 2019 arxiv https://arxiv.org/pdf/1911.00357
#### (9), Auxiliary tasks speed up learning point goal navigation. 2021 CoRL https://arxiv.org/pdf/2007.04561
#### (10), Learning to learn how to learn: Self-adaptive visual navigation using meta-learning. 2019 CVPR https://arxiv.org/pdf/1812.00971
#### (11), Vision-Language Navigation with Self-Supervised Auxiliary Reasoning Tasks 2020,arxiv https://arxiv.org/pdf/1911.07883
#### (12), CLIP-Nav: Using CLIP for Zero-Shot Vision-and-Language Navigation. 2022 arxiv https://arxiv.org/pdf/2211.16649
#### (13), L3MVN: Leveraging Large Language Models for Visual Target Navigation 2022,arxiv https://arxiv.org/pdf/2304.05501
#### (14), NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models. 2023 arxiv https://arxiv.org/pdf/2305.16986

##  三, Embodied-QA (Navigation + Conventional VQA)

###  1, Multi-target EQA
###  2, Multi-agent EQA
###  3, Knowledge Enhancement

#### (1), Embodied Question Answering. 2018 CVPR  https://arxiv.org/pdf/1711.11543
#### (2), Robust-EQA: Robust Learning for Embodied Question Answering With Noisy Labels. 2023 ITNNLS https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10068177
#### (3), Walking with MIND: Mental Imagery eNhanceD Embodied QA. 2019 ACM MM https://arxiv.org/pdf/1908.01482
#### (4), Embodied Multimodal Multitask Learning 2019 IJCAI https://arxiv.org/pdf/1902.01385
#### (5), Multi-Target Embodied Question Answering. 2019 CVPR https://arxiv.org/pdf/1904.04686
#### (6), Multi-agent Embodied Question Answering in Interactive Environments. 2020 ECCV https://link.springer.com/content/pdf/10.1007/978-3-030-58601-0_39.pdf?pdf=inline+link
#### (7), Knowledge-Based Embodied Question Answering. 2021 TPAMI  https://arxiv.org/pdf/2109.07872

# 具身执行 (Embodied Execution):

## 一, Imitation Learning

### 1, Explicit Policy

#### (1), BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning. 2021 CoRL https://arxiv.org/pdf/2202.02005
#### (2), RT-1: Robotics Transformer for Real-World Control at Scale. 2022 Arxiv https://arxiv.org/pdf/2212.06817
#### (3), Behavior Transformers: Cloning k modes with one stone. 2022 NIPS  https://arxiv.org/pdf/2206.11251

### 2, Implicit Policy
#### (1), Implicit Behavioral Cloning. CoRL 2021 https://arxiv.org/pdf/2109.00137

### 3, Diffusion Policy
#### (1), Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023 https://arxiv.org/pdf/2303.04137
#### (2), The Surprising Effectiveness of Representation Learning for Visual Imitation. RSS 2022 https://arxiv.org/pdf/2112.01511

## 二, Reinforcement Learning
### 1, Model-free RL
#### (1), Robotic Grasping using Deep Reinforcement Learning.2020 CoRR  https://arxiv.org/pdf/2007.04499
### 2, Model-based RL
#### (1), TD-MPC2: Scalable, Robust World Models for Continuous Control.2024 ICLR https://arxiv.org/pdf/2310.16828
#### (2), EUCLID: Towards Efficient Unsupervised Reinforcement Learning with Multi-choice Dynamics Model.2023 ICLR https://arxiv.org/pdf/2210.00498

## 三, VLA-based Execution
### 1, End-to-end
#### (1), OpenVLA: An Open-Source Vision-Language-Action Model; Arxiv 24’06; Stanford  https://arxiv.org/pdf/2406.09246
#### (2), RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control; Arxiv 23’07; Google https://arxiv.org/pdf/2307.15818
#### (3), Octo: Octo: An Open-Source Generalist Robot Policy; Arxiv 24’05; UCB https://arxiv.org/pdf/2405.12213
#### (4), π_0: A Vision-Language-Action Flow Model for General Robot Control; Arxiv 24’10; Physical Intelligence https://arxiv.org/pdf/2410.24164
#### (5), CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation; Arxiv 24’11; THU https://arxiv.org/pdf/2411.19650
#### (6), Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression; Arxiv 24’12; ECNU https://arxiv.org/pdf/2412.03293
#### (7), NaVILA: Legged Robot Vision-Language-Action Model for Navigation; Arxiv 24’12; UCSD https://arxiv.org/pdf/2412.04453
#### (8), DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control; Arxiv 25’02; Midea Group https://arxiv.org/pdf/2502.05855
#### (9), SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters; Arxiv 24’12; NTU https://arxiv.org/pdf/2412.00174
#### (10), RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation https://arxiv.org/pdf/2406.04339

### 2, Modular
#### (1), DaDu-E: Rethinking the Role of Large Language Model in Robotic Computing Pipeline; Arxiv 24’12; ICTCAS; https://arxiv.org/pdf/2412.01663
#### (2), ROS-LLM: A ROS framework for embodied AI with task feedback and structured reasoning; Arxiv 24’06; Huawei https://arxiv.org/pdf/2406.19741
#### (3), Closed-Loop Open-Vocabulary Mobile Manipulation with GPT-4V; Arxiv 24’04; BIGAI; https://arxiv.org/pdf/2404.10220v1
#### (4), VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models; CoRL 23; Stanford https://arxiv.org/pdf/2307.05973
#### (5), Code as Policies: Language Model Programs for Embodied Control; Arxiv 22’09; Google; https://arxiv.org/pdf/2209.07753

# 全面优化 (Overall Optimization VLA):

## 一, Network Improvement

### 1, Dynamic Inference
#### (1), DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution; NeurIPS 24’; THU https://arxiv.org/pdf/2411.02359
### 2, MoE
#### (1), ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model; arXiv 25 https://arxiv.org/pdf/2502.14420
#### (2), DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control; arXiv 24, Media https://arxiv.org/pdf/2502.05855

## 二, Performance Improvement

### 1, CoT
#### (1), Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning; Arxiv 24’12; Singapore University of Technology and Design https://arxiv.org/pdf/2412.11974
#### (2), Robotic Control via Embodied Chain-of-Thought Reasoning; Arxiv 24’07; UCB https://arxiv.org/pdf/2407.08693
#### (3), EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought; Arxiv 23’05; HKU https://arxiv.org/pdf/2305.15021

### 2, RL-based Fine-tuning
#### (1), Improving Vision-Language-Action Model with Online Reinforcement Learning; Arxiv,25;THU https://arxiv.org/pdf/2501.16664
#### (2), VLM-RL: A Unified Vision Language Models and Reinforcement Learning Framework for Safe Autonomous Driving; Arxiv 25; University of Wisconsin-Madison https://arxiv.org/pdf/2412.15544

## 三, Computing Optimization
### 1, Quantization
#### (1), MBQ: Modality-Balanced Quantization for Large Vision-Language Models; Arxiv 24’12; THU https://arxiv.org/pdf/2412.19509
#### (2), Quantization-Aware Imitation-Learning for Resource-Efficient Robotic Control; Arxiv 24’12; Hanyang University https://arxiv.org/pdf/2412.01034

### 2, Cache
#### (1), MBQ: Modality-Balanced Quantization for Large Vision-Language Models; Arxiv 24’12; THU https://arxiv.org/pdf/2412.19509
#### (2), Quantization-Aware Imitation-Learning for Resource-Efficient Robotic Control; Arxiv 24’12; Hanyang University https://arxiv.org/pdf/2412.01034
#### (3), VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation; 25’02; University of Sydney https://arxiv.org/pdf/2502.02175

### 3, DP Distillation
#### (1), One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation; Arxiv 24’10; NVIDIA https://arxiv.org/pdf/2410.21257
#### (2), Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation; Arxiv 24’05; Stanford https://arxiv.org/pdf/2405.07503

## 四, System Design
### 1, Co-design
#### (1), Software-Hardware Co-Design For Embodied AI Robots; Arxiv 24’07; ICTCAS https://arxiv.org/pdf/2407.04292
### 2, Parallel Strategy
#### (1), A Dual Process VLA: Efficient Robotic Manipulation Leveraging VLM; Arxiv 25’03; Republic of Korea https://arxiv.org/pdf/2410.15549
### 3, Resource Allocation
#### (1), A Dual Process VLA: Efficient Robotic Manipulation Leveraging VLM; Arxiv 25’03; Republic of Korea https://arxiv.org/pdf/2410.15549
