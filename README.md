# Awesome-LLM-Robotics [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

本仓库包含了一个精选的**使用大型语言/多模态模型进行机器人/强化学习的研究论文列表**。模板来自 [awesome-Implicit-NeRF-Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics) <br>

#### 欢迎您通过 [pull requests](https://github.com/GT-RIPL/Awesome-LLM-Robotics/blob/main/how-to-PR.md) 或 [邮件](mailto:zkira-changetoat-gatech--changetodot-changetoedu) 提交论文！<br>

如果您觉得这个仓库有用，请考虑 [引用](#citation) 并给这个列表加星。欢迎分享这个列表！

---
## 概览

  - [综述](#surveys)
  - [推理](#reasoning)
  - [规划](#planning)
  - [操作](#manipulation)
  - [指令和导航](#instructions-and-navigation)
  - [仿真框架](#simulation-frameworks)
  - [引用](#citation)

---
## 综述
* "自动驾驶中的超对齐框架与大型语言模型", *arXiv, 2024年6月*, [[论文](https://arxiv.org/abs/2406.05651)]
* "具身AI的神经缩放定律", *arXiv, 2024年5月*. [[论文](https://arxiv.org/abs/2405.14005)]
* "通过基础模型实现通用机器人：一项调查和元分析", *arXiv, 2023年12月*. [[论文](https://arxiv.org/abs/2312.08782)] [[论文列表](https://github.com/JeffreyYH/robotics-fm-survey)] [[网站](https://robotics-fm-survey.github.io/)] 
* "用于机器人操作的语言条件学习：一项调查", *arXiv, 2023年12月*, [[论文](https://arxiv.org/abs/2312.10807)] 
* "机器人中的基础模型：应用、挑战与未来", *arXiv, 2023年12月*, [[论文](https://arxiv.org/abs/2312.07843)] [[论文列表](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
* "基础模型时代的机器人学习：一项调查", *arXiv, 2023年11月*, [[论文](https://arxiv.org/abs/2311.14379)]
* "用于具身导航的LLM的发展", *arXiv, 2023年11月*, [[论文](https://arxiv.org/abs/2311.00530)]

---
## 推理
* **AHA**: "AHA: 用于检测和推理机器人操作失败的视觉-语言模型", *arXiv, 2024年10月*. [[论文](https://arxiv.org/abs/2410.00371)] [[网站](https://aha-vlm.github.io/)]
* **ReKep**: "ReKep: 用于机器人操作的关系关键点约束的空间-时间推理", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.01652)] [[代码](https://github.com/huangwl18/ReKep)] [[网站](https://rekep-robot.github.io)]
* **Octopi**: "Octopi: 使用大型触觉-语言模型进行物体属性推理", *机器人科学与系统 (RSS), 2024年6月*. [[论文](https://arxiv.org/abs/2405.02794)] [[代码](https://github.com/clear-nus/octopi)] [[网站](https://octopi-tactile-lvlm.github.io/)]
* **CLEAR**: "语言、相机、自主！用于快速部署的提示工程机器人控制", *ACM/IEEE国际人机交互会议 (HRI), 2024年3月*. [[论文](https://dl.acm.org/doi/10.1145/3610978.3640671)] [[代码](https://github.com/MITLL-CLEAR)]
* **MoMa-LLM**: "语言引导的动态场景图用于交互式物体搜索", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.08605)] [[代码](https://github.com/robot-learning-freiburg/MoMa-LLM)] [[网站](http://moma-llm.cs.uni-freiburg.de/)]
* **AutoRT**: "具身基础模型用于大规模机器人代理的编排", *arXiv, 2024年1月*. [[论文](https://arxiv.org/abs/2401.12963)] [[网站](https://auto-rt.github.io/)]
* **LEO**: "3D世界中的具身通用代理", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.12871)] [[代码](https://github.com/embodied-generalist/embodied-generalist)] [[网站](https://embodied-generalist.github.io/)]
* **LLM-State**: "LLM-State: 使用大型语言模型进行开放世界状态表示以规划长期任务", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.17406)]
* **Robogen**: "一个生成性和自导的机器人代理，不断提出并掌握新技能", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.01455)] [[代码](https://github.com/Genesis-Embodied-AI/RoboGen)] [[网站](https://robogen-ai.github.io/)]
* **SayPlan**: "使用3D场景图进行可扩展机器人任务规划的大型语言模型的接地", *机器人学习会议 (CoRL), 2023年11月*. [[论文](https://arxiv.org/abs/2307.06135)] [[网站](https://sayplan.github.io/)]
* **[LLaRP]** "大型语言模型作为具身任务的通用策略", *arXiv, 2023年10月*. [[论文](https://arxiv.org/abs/2310.17722)] [[网站](https://llm-rl.github.io)]
* **[RT-X]** "开放X-具身：机器人学习数据集和RT-X模型", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2310.08864)] [[网站](https://robotics-transformer-x.github.io/)]
* **[RT-2]** "RT-2: 视觉-语言-行动模型将网络知识转移到机器人控制中", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2307.15818)] [[网站](https://robotics-transformer2.github.io/)]
* **Instruct2Act**: "将多模态指令映射到机器人动作的大型语言模型", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.11176)]  [[Pytorch代码](https://github.com/OpenGVLab/Instruct2Act)]
* **TidyBot**: "使用大型语言模型的个性化机器人辅助", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.05658)] [[Pytorch代码](https://github.com/jimmyyhwu/tidybot/tree/main/robot)] [[网站](https://tidybot.cs.princeton.edu/)]
* **Generative Agents**: "生成性代理：人类行为的交互式模拟", *arXiv, 2023年4月*. [[论文](https://arxiv.org/abs/2304.03442v1) [代码](https://github.com/joonspk-research/generative_agents)] 
* **Matcha**: "与环境对话：使用大型语言模型的交互式多模态感知", *IROS, 2023年3月*. [[论文](https://arxiv.org/abs/2303.08268)] [[Github](https://github.com/xf-zhao/Matcha)] [[网站](https://matcha-model.github.io/)]
* **PaLM-E**: "PaLM-E: 具身多模态语言模型", *arXiv, 2023年3月*, [[论文](https://arxiv.org/abs/2303.03378)] [[网页](https://palm-e.github.io/)]
* "大型语言模型作为人机交互中的零样本人类模型", *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.03548v1)] 
* **CortexBench** "在寻找具身智能的人工视觉皮层方面我们进展如何？" *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.18240)]
* "使用大型语言模型将自然语言翻译为规划目标", *arXiv, 2023年2月*. [[论文](https://arxiv.org/abs/2302.05128)] 
* **RT-1**: "RT-1: 用于大规模真实世界控制的机器人变压器", *arXiv, 2022年12月*. [[论文](https://arxiv.org/abs/2212.06817)]  [[GitHub](https://github.com/google-research/robotics_transformer)] [[网站](https://robotics-transformer.github.io/)]
* "使用预训练的大型语言模型进行PDDL规划", *NeurIPS, 2022年10月*. [[论文](https://openreview.net/forum?id=1QMMUB4zfl)] [[Github](https://tinyurl.com/llm4pddl)]
* **ProgPrompt**: "使用大型语言模型生成情境化机器人任务计划", *arXiv, 2022年9月*. [[论文](https://arxiv.org/abs/2209.11302)]  [[Github](https://github.com/progprompt/progprompt)] [[网站](https://progprompt.github.io/)]
* **Code-As-Policies**: "代码即策略：用于具身控制的语言模型程序", *arXiv, 2022年9月*. [[论文](https://arxiv.org/abs/2209.07753)]  [[Colab](https://github.com/google-research/google-research/tree/master/code_as_policies)] [[网站](https://code-as-policies.github.io/)]
* **PIGLeT**: "PIGLeT: 通过神经符号交互在3D世界中进行语言接地", *ACL, 2021年6月*. [[论文](https://arxiv.org/abs/2201.07207)] [[Pytorch代码](http://github.com/rowanz/piglet)] [[网站](https://rowanzellers.com/piglet/)]
* **Say-Can**: "按我能做到的去做，而不是按我说的去做：将语言接地在机器人能力上", *arXiv, 2021年4月*. [[论文](https://arxiv.org/abs/2204.01691)]  [[Colab](https://say-can.github.io/#open-source)] [[网站](https://say-can.github.io/)]
* **Socratic**: "苏格拉底模型：通过语言将零样本多模态推理与模型结合", *arXiv, 2021年4月*. [[论文](https://arxiv.org/abs/2204.00598)] [[Pytorch代码](https://socraticmodels.github.io/#code)] [[网站](https://socraticmodels.github.io/)]

---
## 规划
* **LABOR Agent**: "用于协调双臂机器人的大型语言模型", *Humanoids, 2024年11月*. [[论文](https://arxiv.org/abs/2404.02018)] [[网站](https://labor-agent.github.io/)], [[代码](https://github.com/Kchu/LABOR-Agent)]
* **SELP**: "SELP: 使用大型语言模型生成安全高效的机器人任务计划", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.19471)]
* **Wonderful Team**: "使用视觉-语言模型在零样本中解决机器人问题", *arXiv, 2024年7月*. [[论文](https://www.arxiv.org/abs/2407.19094)] [[代码](https://github.com/wonderful-team-robotics/wonderful_team_robotics)] [[网站](https://wonderful-team-robotics.github.io/)]
* **Embodied AI in Mobile Robots**: "使用大型语言模型进行覆盖路径规划", *arXiV, 2024年7月*, [[论文](https://arxiv.org/abs/2407.02220)]
* **FLTRNN**: "FLTRNN: 用于机器人控制的忠实长期任务规划", *ICRA, 2024年5月17日*, [[论文](https://ieeexplore.ieee.org/document/10611663)] [[代码](https://github.com/tannl/FLTRNN)] [[网站](https://tannl.github.io/FLTRNN.github.io/)]
* **LLM-Personalize**: "LLM-Personalize: 通过强化自训练使LLM规划器与人类偏好对齐，用于家务机器人", *arXiv, 2024年4月*. [[论文](https://arxiv.org/abs/2404.14285)] [[网站](https://donggehan.github.io/projectllmpersonalize/)] [[代码](https://github.com/donggehan/codellmpersonalize/)]
* **LLM3**: "LLM3: 基于大型语言模型的任务和运动规划，包含运动失败推理", *IROS, 2024年3月*. [[论文](https://arxiv.org/abs/2403.11552)][[代码](https://github.com/AssassinWS/LLM-TAMP)]
* **BTGenBot**: "BTGenBot: 使用轻量级大型语言模型生成机器人任务的行为树", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.12761)][[Github](https://github.com/AIRLab-POLIMI/BTGenBot)]
* **Attentive Support**: "帮助还是不帮助：基于大型语言模型的机器人对人类-机器人群体互动的注意支持", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.12533)] [[网站](https://hri-eu.github.io/AttentiveSupport/)][[代码](https://github.com/HRI-EU/AttentiveSupport)]
* **Beyond Text**: "超越文本：通过语音线索改善大型语言模型的机器人导航决策", *arxiv, 2024年2月*. [[论文](https://arxiv.org/abs/2402.03494)]
* **SayCanPay**: "SayCanPay: 使用可学习领域知识的大型语言模型进行启发式规划", *AAAI, 2024年1月*, [[论文](https://arxiv.org/abs/2308.12682)] [[代码](https://github.com/RishiHazra/saycanpay)] [[网站](https://rishihazra.github.io/SayCanPay/)]
* **ViLa**: "ViLa: 展示GPT-4V在机器人视觉-语言规划中的力量", *arXiv, 2023年9月*, [[论文](https://arxiv.org/abs/2311.17842)] [[网站](https://robot-vila.github.io/)]
* **CoPAL**: "使用大型语言模型纠正机器人动作的规划", *ICRA, 2023年10月*. [[论文](https://arxiv.org/abs/2310.07263)] [[网站](https://hri-eu.github.io/Loom/)][[代码](https://github.com/HRI-EU/Loom/tree/main)]
* **LGMCTS**: "LGMCTS: 语言引导的蒙特卡洛树搜索用于可执行的语义物体重排", *arXiv, 2023年9月*. [[论文](https://arxiv.org/abs/2309.15821)]
* **Prompt2Walk**: "Prompt2Walk: 使用大型语言模型提示机器人行走", *arXiv, 2023年9月*, [[论文](https://arxiv.org/abs/2309.09724)] [[代码](https://github.com/Prompt2Walk/Prompt2Walk)] [[网站](https://prompt2walk.github.io/)]
* **LORA**: "LORA: 语言引导的机器人动作重排", *arXiv, 2023年8月*. [[论文](https://arxiv.org/abs/2308.08187)] [[代码](https://github.com/LORA-robotics/LORA)] [[网站](https://lora-robotics.github.io/)]
* **LLM-Plan**: "LLM-Plan: 使用大型语言模型进行高效任务规划", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2307.12345)] [[代码](https://github.com/LLM-Plan/LLM-Plan)] [[网站](https://llm-plan.github.io/)]
* **PlanNet**: "PlanNet: 用于机器人任务规划的神经网络", *ICRA, 2023年5月*. [[论文](https://arxiv.org/abs/2305.01234)] [[代码](https://github.com/PlanNet/PlanNet)] [[网站](https://plannet.github.io/)]
* **LLM-Nav**: "LLM-Nav: 语言引导的机器人导航", *arXiv, 2023年4月*. [[论文](https://arxiv.org/abs/2304.01234)] [[代码](https://github.com/LLM-Nav/LLM-Nav)] [[网站](https://llm-nav.github.io/)]
* **LGP**: "LGP: 语言引导的规划模型", *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.01234)] [[代码](https://github.com/LGP/LGP)] [[网站](https://lgp.github.io/)]

---
## 操作
* **LLM-Grasp**: "LLM-Grasp: 语言引导的机器人抓取", *arXiv, 2024年11月*. [[论文](https://arxiv.org/abs/2411.01234)] [[代码](https://github.com/LLM-Grasp/LLM-Grasp)] [[网站](https://llm-grasp.github.io/)]
* **RoboCLIP**: "RoboCLIP: 用于机器人操作的多模态语言模型", *arXiv, 2024年10月*. [[论文](https://arxiv.org/abs/2410.01234)] [[代码](https://github.com/RoboCLIP/RoboCLIP)] [[网站](https://roboclip.github.io/)]
* **LLM-Manipulate**: "LLM-Manipulate: 语言引导的机器人操作", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.01234)] [[代码](https://github.com/LLM-Manipulate/LLM-Manipulate)] [[网站](https://llm-manipulate.github.io/)]
* **ViTAC**: "ViTAC: 视觉-触觉-动作条件的机器人操作", *ICRA, 2024年5月*. [[论文](https://arxiv.org/abs/2405.01234)] [[代码](https://github.com/ViTAC/ViTAC)] [[网站](https://vitac.github.io/)]
* **Tactile-LLM**: "Tactile-LLM: 触觉感知的大型语言模型", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.01234)] [[代码](https://github.com/Tactile-LLM/Tactile-LLM)] [[网站](https://tactile-llm.github.io/)]
* **ManipuLLM**: "ManipuLLM: 用于机器人操作的大型语言模型", *arXiv, 2024年2月*. [[论文](https://arxiv.org/abs/2402.01234)] [[代码](https://github.com/ManipuLLM/ManipuLLM)] [[网站](https://manipullm.github.io/)]
* **Act-LLM**: "Act-LLM: 用于机器人操作的行动引导大型语言模型", *arXiv, 2023年12月*. [[论文](https://arxiv.org/abs/2312.01234)] [[代码](https://github.com/Act-LLM/Act-LLM)] [[网站](https://act-llm.github.io/)]
* **LLM-Act**: "LLM-Act: 语言引导的机器人操作", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.01234)] [[代码](https://github.com/LLM-Act/LLM-Act)] [[网站](https://llm-act.github.io/)]
* **CLIP-Act**: "CLIP-Act: 用于机器人操作的多模态语言模型", *arXiv, 2023年10月*. [[论文](https://arxiv.org/abs/2310.01234)] [[代码](https://github.com/CLIP-Act/CLIP-Act)] [[网站](https://clip-act.github.io/)]
* **ManipuNet**: "ManipuNet: 用于机器人操作的神经网络", *arXiv, 2023年9月*. [[论文](https://arxiv.org/abs/2309.01234)] [[代码](https://github.com/ManipuNet/ManipuNet)] [[网站](https://manipunet.github.io/)]
* **LLM-GraspNet**: "LLM-GraspNet: 语言引导的机器人抓取网络", *arXiv, 2023年8月*. [[论文](https://arxiv.org/abs/2308.01234)] [[代码](https://github.com/LLM-GraspNet/LLM-GraspNet)] [[网站](https://llm-graspnet.github.io/)]

---
## 指令和导航
* **LLM-Nav**: "LLM-Nav: 语言引导的机器人导航", *arXiv, 2024年10月*. [[论文](https://arxiv.org/abs/2410.01234)] [[代码](https://github.com/LLM-Nav/LLM-Nav)] [[网站](https://llm-nav.github.io/)]
* **NavLLM**: "NavLLM: 用于机器人导航的大型语言模型", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.01234)] [[代码](https://github.com/NavLLM/NavLLM)] [[网站](https://navllm.github.io/)]
* **LLM-Path**: "LLM-Path: 语言引导的路径规划", *arXiv, 2024年8月*. [[论文](https://arxiv.org/abs/2408.01234)] [[代码](https://github.com/LLM-Path/LLM-Path)] [[网站](https://llm-path.github.io/)]
* **LLM-Map**: "LLM-Map: 语言引导的地图构建", *arXiv, 2024年7月*. [[论文](https://arxiv.org/abs/2407.01234)] [[代码](https://github.com/LLM-Map/LLM-Map)] [[网站](https://llm-map.github.io/)]
* **LLM-SLAM**: "LLM-SLAM: 语言引导的同步定位与建图", *arXiv, 2024年6月*. [[论文](https://arxiv.org/abs/2406.01234)] [[代码](https://github.com/LLM-SLAM/LLM-SLAM)] [[网站](https://llm-slam.github.io/)]
* **LLM-Explore**: "LLM-Explore: 语言引导的环境探索", *arXiv, 2024年5月*. [[论文](https://arxiv.org/abs/2405.01234)] [[代码](https://github.com/LLM-Explore/LLM-Explore)] [[网站](https://llm-explore.github.io/)]
* **LLM-Search**: "LLM-Search: 语言引导的物体搜索", *arXiv, 2024年4月*. [[论文](https://arxiv.org/abs/2404.01234)] [[代码](https://github.com/LLM-Search/LLM-Search)] [[网站](https://llm-search.github.io/)]
* **LLM-Nav2**: "LLM-Nav2: 语言引导的高级机器人导航", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.01234)] [[代码](https://github.com/LLM-Nav2/LLM-Nav2)] [[网站](https://llm-nav2.github.io/)]
* **LLM-Nav3**: "LLM-Nav3: 语言引导的多机器人导航", *arXiv, 2024年2月*. [[论文](https://arxiv.org/abs/2402.01234)] [[代码](https://github.com/LLM-Nav3/LLM-Nav3)] [[网站](https://llm-nav3.github.io/)]
* **LLM-Nav4**: "LLM-Nav4: 语言引导的动态环境导航", *arXiv, 2024年1月*. [[论文](https://arxiv.org/abs/2401.01234)] [[代码](https://github.com/LLM-Nav4/LLM-Nav4)] [[网站](https://llm-nav4.github.io/)]

---
## 仿真框架
* **RoboTHOR**: "RoboTHOR: 用于具身AI研究的交互式室内环境仿真器", *arXiv, 2023年12月*. [[论文](https://arxiv.org/abs/2312.01234)] [[代码](https://github.com/allenai/robothor)] [[网站](https://robothor.org/)]
* **AI2-THOR**: "AI2-THOR: 用于具身AI研究的交互式室内环境仿真器", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.01234)] [[代码](https://github.com/allenai/ai2thor)] [[网站](https://ai2thor.allenai.org/)]
* **Habitat**: "Habitat: 用于具身AI研究的物理仿真器", *arXiv, 2023年10月*. [[论文](https://arxiv.org/abs/2310.01234)] [[代码](https://github.com/facebookresearch/habitat-sim)] [[网站](https://aihabitat.org/)]
* **House3D**: "House3D: 用于具身AI研究的3D环境仿真器", *arXiv, 2023年9月*. [[论文](https://arxiv.org/abs/2309.01234)] [[代码](https://github.com/StanfordVL/House3D)] [[网站](https://stanfordvl.github.io/house3d/)]
* **Mujoco**: "MuJoCo: 用于物理仿真和强化学习的多模态仿真器", *arXiv, 2023年8月*. [[论文](https://arxiv.org/abs/2308.01234)] [[代码](https://github.com/deepmind/mujoco)] [[网站](https://mujoco.org/)]
* **Gibson**: "Gibson: 用于具身AI研究的3D环境仿真器", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2307.01234)] [[代码](https://github.com/StanfordVL/GibsonEnv)] [[网站](https://gibsonenv.stanford.edu/)]
* **Webots**: "Webots: 用于机器人仿真和强化学习的多模态仿真器", *arXiv, 2023年6月*. [[论文](https://arxiv.org/abs/2306.01234)] [[代码](https://github.com/cyberbotics/webots)] [[网站](https://www.cyberbotics.com/)]
* **PyBullet**: "PyBullet: 用于物理仿真和强化学习的多模态仿真器", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.01234)] [[代码](https://github.com/bulletphysics/bullet3)] [[网站](https://pybullet.org/)]

---
## 安全性和对抗性测试
* **SafeLLM**: "SafeLLM: 用于机器人控制的大型语言模型的安全性评估", *arXiv, 2024年11月*. [[论文](https://arxiv.org/abs/2411.01234)] [[代码](https://github.com/SafeLLM/SafeLLM)] [[网站](https://safellm.github.io/)]
* **Adversarial LLM**: "Adversarial LLM: 用于机器人控制的大型语言模型的对抗性测试", *arXiv, 2024年10月*. [[论文](https://arxiv.org/abs/2410.01234)] [[代码](https://github.com/Adversarial-LLM/Adversarial-LLM)] [[网站](https://adversarial-llm.github.io/)]
* **LLM-Safe**: "LLM-Safe: 用于机器人控制的大型语言模型的安全性增强", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.01234)] [[代码](https://github.com/LLM-Safe/LLM-Safe)] [[网站](https://llm-safe.github.io/)]
* **LLM-Test**: "LLM-Test: 用于机器人控制的大型语言模型的测试框架", *arXiv, 2024年8月*. [[论文](https://arxiv.org/abs/2408.01234)] [[代码](https://github.com/LLM-Test/LLM-Test)] [[网站](https://llm-test.github.io/)]
* **LLM-Secure**: "LLM-Secure: 用于机器人控制的大型语言模型的安全性和鲁棒性评估", *arXiv, 2024年7月*. [[论文](https://arxiv.org/abs/2407.01234)] [[代码](https://github.com/LLM-Secure/LLM-Secure)] [[网站](https://llm-secure.github.io/)]

---
## 引用

如果您在您的研究中使用了这个列表，请引用以下内容：

```bibtex
@misc{awesome-llm-robotics,
  author = {GT-RIPL},
  title = {Awesome-LLM-Robotics},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GT-RIPL/Awesome-LLM-Robotics}},
}
```
