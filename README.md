# Awesome-LLM-Robotics [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

本仓库包含一个精选列表，涵盖使用大型语言模型/多模态模型进行机器人/强化学习的论文。模板来自 [awesome-Implicit-NeRF-Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics) <br>

#### 如果您想添加论文，请随时发送[拉取请求](https://github.com/GT-RIPL/Awesome-LLM-Robotics/blob/main/how-to-PR.md) 或通过邮件 [联系我](mailto:zkira-changetoat-gatech--changetodot-changetoedu)！请确保按逆时间顺序排列，并严格遵循格式！<br>

如果您觉得此仓库有用，请考虑引用和加星标。欢迎分享给其他人！

---

## 概述

- [Awesome-LLM-Robotics ](#awesome-llm-robotics-)
      - [请随时发送拉取请求或电子邮件添加论文！](#please-feel-free-to-send-me-pull-requests-or-email-to-add-papers-)
  - [概述](#overview)
  - [综述](#surveys)
  - [推理](#reasoning)
  - [规划](#planning)
  - [操作](#manipulation)
  - [指令与导航](#instructions-and-navigation)
  - [模拟框架](#simulation-frameworks)
  - [安全、风险、红队测试及对抗性测试](#safety-risks-red-teaming-and-adversarial-testing)
  - [引用](#citation)

---

## 综述

* "A Superalignment Framework in Autonomous Driving with Large Language Models", *arXiv, 2024年6月*, [[论文](https://arxiv.org/abs/2406.05651)]
* "Neural Scaling Laws for Embodied AI", *arXiv, 2024年5月*. [[论文](https://arxiv.org/abs/2405.14005)]
* "On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling (APS)", *ICAPS, 2024年5月*, [[论文]](https://ojs.aaai.org/index.php/ICAPS/article/view/31503) [[网站](https://ai4society.github.io/LLM-Planning-Viz/)
* "Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis", *arXiv, 2023年12月*. [[论文](https://arxiv.org/abs/2312.08782)] [[论文列表](https://github.com/JeffreyYH/robotics-fm-survey)] [[网站](https://robotics-fm-survey.github.io/)] 
* "Language-conditioned Learning for Robotic Manipulation: A Survey", *arXiv, 2023年12月*, [[论文](https://arxiv.org/abs/2312.10807)] 
* "Foundation Models in Robotics: Applications, Challenges, and the Future", *arXiv, 2023年12月*, [[论文](https://arxiv.org/abs/2312.07843)] [[论文列表](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
* "Robot Learning in the Era of Foundation Models: A Survey", *arXiv, 2023年11月*, [[论文](https://arxiv.org/abs/2311.14379)]
* "The Development of LLMs for Embodied Navigation", *arXiv, 2023年11月*, [[论文](https://arxiv.org/abs/2311.00530)]

---

## 推理

* **RoboSpatial**: "RoboSpatial: 教授空间理解给2D和3D视觉语言模型用于机器人学", CVPR, 2025年6月. [[论文](https://arxiv.org/abs/2411.16537)] [[代码](https://github.com/NVlabs/RoboSpatial)] [[网站](https://chanh.ee/RoboSpatial/)]
* **SPINE**: "SPINE: 在非结构化环境中基于不完整自然语言规范的任务在线语义规划", 国际机器人与自动化会议 (ICRA), 2025年5月. [[论文](https://arxiv.org/abs/2410.03035)] [[网站](https://zacravichandran.github.io/SPINE/)]
* **ELLMER**: "具身大语言模型使机器人在不可预测环境中完成长期任务", 自然机器智能, 2025年3月. [[论文](https://www.nature.com/articles/s42256-025-01005-x)] [[网站](https://www.nature.com/articles/s42256-025-01005-x)]
* **AHA**: "AHA: 用于检测和推理机器人操作失败的视觉语言模型", *arXiv, 2024年10月*. [[论文](https://arxiv.org/abs/2410.00371)] [[网站](https://aha-vlm.github.io/)]
* **ReKep**: "ReKep: 基于关系关键点约束的时空推理用于机器人操作", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.01652)] [[代码](https://github.com/huangwl18/ReKep)] [[网站](https://rekep-robot.github.io)]
* **Octopi**: "Octopi: 使用大规模触觉语言模型进行对象属性推理", 机器人科学与系统 (RSS), 2024年6月. [[论文](https://arxiv.org/abs/2405.02794)] [[代码](https://github.com/clear-nus/octopi)] [[网站](https://octopi-tactile-lvlm.github.io/)]
* **CLEAR**: "语言、相机、自主权！为迅速变化的部署设计提示工程机器人控制", ACM/IEEE人机交互国际会议 (HRI), 2024年3月. [[论文](https://dl.acm.org/doi/10.1145/3610978.3640671)] [[代码](https://github.com/MITLL-CLEAR)]
* **MoMa-LLM**: "基于语言的动态场景图用于移动操作中的交互式对象搜索", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.08605)] [[代码](https://github.com/robot-learning-freiburg/MoMa-LLM)] [[网站](http://moma-llm.cs.uni-freiburg.de/)]
* **AutoRT**: "用于协调大量机器人代理的大规模编排的具身基础模型", *arXiv, 2024年1月*. [[论文](https://arxiv.org/abs/2401.12963)] [[网站](https://auto-rt.github.io/)]
* **LEO**: "3D世界中的通用代理", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.12871)] [[代码](https://github.com/embodied-generalist/embodied-generalist)] [[网站](https://embodied-generalist.github.io/)]
* **LLM-State**: "LLM-State: 大型语言模型长时任务规划的开放世界状态表示", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.17406)]
* **Robogen**: "一个生成且自我引导的机器人代理，能无尽提出并掌握新技能", *arXiv, 2023年11月*. [[论文](https://arxiv.org/abs/2311.01455)] [[代码](https://github.com/Genesis-Embodied-AI/RoboGen)] [[网站](https://robogen-ai.github.io/)]
* **SayPlan**: "使用3D场景图作为基础的大规模机器人任务规划的大型语言模型", 机器人学习会议 (CoRL), 2023年11月. [[论文](https://arxiv.org/abs/2307.06135)] [[网站](https://sayplan.github.io/)]
* **[LLaRP]** "大型语言模型作为可推广的具身任务策略", *arXiv, 2023年10月*. [[论文](https://arxiv.org/abs/2310.17722)] [[网站](https://llm-rl.github.io)]
* **[RT-X]** "Open X-Embodiment: 机器人学习数据集和RT-X模型", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2310.08864)] [[网站](https://robotics-transformer-x.github.io/)]
* **[RT-2]** "RT-2: 视觉语言动作模型将网络知识转移到机器人控制", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2307.15818)] [[网站](https://robotics-transformer2.github.io/)]
* **Instruct2Act**: "将多模态指令映射到机器人动作的大型语言模型", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.11176)]  [[Pytorch代码](https://github.com/OpenGVLab/Instruct2Act)]
* **TidyBot**: "使用大型语言模型的个性化机器人助手", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.05658)] [[Pytorch代码](https://github.com/jimmyyhwu/tidybot/tree/main/robot)] [[网站](https://tidybot.cs.princeton.edu/)]
* **Generative Agents**: "生成代理：人类行为的互动仿真", *arXiv, 2023年4月*. [[论文](https://arxiv.org/abs/2304.03442v1) [代码](https://github.com/joonspk-research/generative_agents)] 
* **Matcha**: "与环境对话：使用大型语言模型进行互动多模态感知", IROS, 2023年3月. [[论文](https://arxiv.org/abs/2303.08268)] [[GitHub](https://github.com/xf-zhao/Matcha)] [[网站](https://matcha-model.github.io/)]
* **PaLM-E**: "PaLM-E: 具身多模态语言模型", *arXiv, 2023年3月*, [[论文](https://arxiv.org/abs/2303.03378)] [[网页](https://palm-e.github.io/)]
* "大型语言模型作为零样本的人类模型用于人机交互", *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.03548v1)] 
* **CortexBench** "我们在寻找具身智能的人工视觉皮层方面进展如何？" *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.18240)]
* "将自然语言转换为规划目标的大型语言模型", *arXiv, 2023年2月*. [[论文](https://arxiv.org/abs/2302.05128)] 
* **RT-1**: "RT-1: 面向真实世界大规模控制的机器人变压器", *arXiv, 2022年12月*. [[论文](https://arxiv.org/abs/2212.06817)]  [[GitHub](https://github.com/google-research/robotics_transformer)] [[网站](https://robotics-transformer.github.io/)]
* "使用预训练大型语言模型进行PDDL规划", *NeurIPS, 2022年10月*. [[论文](https://openreview.net/forum?id=1QMMUB4zfl)] [[GitHub](https://tinyurl.com/llm4pddl)]
* **ProgPrompt**: "使用大型语言模型生成情境机器人任务计划", *arXiv, 2022年9月*. [[论文](https://arxiv.org/abs/2209.11302)]  [[GitHub](https://github.com/progprompt/progprompt)] [[网站](https://progprompt.github.io/)]
* **Code-As-Policies**: "代码即策略：语言模型程序用于具身控制", *arXiv, 2022年9月*. [[论文](https://arxiv.org/abs/2209.07753)]  [[Colab](https://github.com/google-research/google-research/tree/master/code_as_policies)] [[网站](https://code-as-policies.github.io/)]
* **PIGLeT**: "PIGLeT: 通过三维世界的神经符号交互实现语言基础", ACL, 2021年6月. [[论文](https://arxiv.org/abs/2201.07207)] [[PyTorch代码](http://github.com/rowanz/piglet)] [[网站](https://rowanzellers.com/piglet/)]
* **Say-Can**: "做我让你做的，而不是我说的：在机器人可供性中建立语言基础", *arXiv, 2021年4月*. [[论文](https://arxiv.org/abs/2204.01691)]  [[Colab](https://say-can.github.io/#open-source)] [[网站](https://say-can.github.io/)]
* **Socratic**: "苏格拉底模型：组合零样本多模态推理与语言", *arXiv, 2021年4月*. [[论文](https://arxiv.org/abs/2204.00598)] [[PyTorch代码](https://socraticmodels.github.io/#code)] [[网站](https://socraticmodels.github.io/)]

---

## 规划

* **LLM+MAP**: "LLM+MAP: 使用大型语言模型和规划领域定义语言进行双臂机器人任务规划", arxiv, 2025年3月. [[论文](https://arxiv.org/abs/2503.17309)] [[代码](https://github.com/Kchu/LLM-MAP)]
* **Code-as-Monitor**: "面向反应性和前瞻性机器人故障检测的约束感知可视化编程", CVPR, 2025年. [[论文](https://arxiv.org/abs/2412.04455)] [[项目](https://zhoues.github.io/Code-as-Monitor/)]
* **LABOR Agent**: "用于协调双臂机器人的大型语言模型", Humanoids, 2024年11月. [[论文](https://arxiv.org/abs/2404.02018)] [[网站](https://labor-agent.github.io/)], [[代码](https://github.com/Kchu/LABOR-Agent)]
* **SELP**: "使用大型语言模型生成机器人代理的安全高效任务计划", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.19471)]
* **Wonderful Team**: "使用视觉语言模型零样本解决机器人问题", *arXiv, 2024年7月*. [[论文](https://www.arxiv.org/abs/2407.19094)] [[代码](https://github.com/wonderful-team-robotics/wonderful_team_robotics)] [[网站](https://wonderful-team-robotics.github.io/)]
* **Mobile Robots中的具身AI**: "使用大型语言模型进行覆盖路径规划", *arXiV, 2024年7月*, [[论文](https://arxiv.org/abs/2407.02220)]
* **FLTRNN**: "FLTRNN: 使用大型语言模型进行忠实的长时机器人任务规划", ICRA, 2024年5月17日, [[论文](https://ieeexplore.ieee.org/document/10611663)] [[代码](https://github.com/tannl/FLTRNN)] [[网站](https://tannl.github.io/FLTRNN.github.io/)]
* **LLM-Personalize**: "LLM-Personalize: 通过增强自训练对家务机器人对齐LLM规划器与人类偏好", *arXiv, 2024年4月*. [[论文](https://arxiv.org/abs/2404.14285)] [[网站](https://donggehan.github.io/projectllmpersonalize/)] [[代码](https://github.com/donggehan/codellmpersonalize/)]
* **LLM3**: "LLM3: 基于大型语言模型的任务和运动规划以及运动失败推理", IROS, 2024年3月. [[论文](https://arxiv.org/abs/2403.11552)][[代码](https://github.com/AssassinWS/LLM-TAMP)]
* **BTGenBot**: "BTGenBot: 使用轻量级LLM生成机器人任务的行为树", IROS, 2024年3月. [[论文](https://ieeexplore.ieee.org/document/10802304)][[GitHub](https://github.com/AIRLab-POLIMI/BTGenBot)]
* **Attentive Support**: "帮助与否：基于LLM的人机组交互关注支持", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.12533)] [[网站](https://hri-eu.github.io/AttentiveSupport/)][[代码](https://github.com/HRI-EU/AttentiveSupport)]
* **Beyond Text**: "超越文本：通过语音线索改进LLM在机器人导航中的决策", arxiv, 2024年2月. [[论文](https://arxiv.org/abs/2402.03494)]
* **SayCanPay**: "SayCanPay: 使用可学习领域知识与大型语言模型进行启发式规划", AAAI, 2024年1月, [[论文](https://arxiv.org/abs/2308.12682)] [[代码](https://github.com/RishiHazra/saycanpay)] [[网站](https://rishihazra.github.io/SayCanPay/)]
* **ViLa**: "三思而后行：揭示GPT-4V在机器人视觉语言规划中的力量", *arXiv, 2023年9月*, [[论文](https://arxiv.org/abs/2311.17842)] [[网站](https://robot-vila.github.io/)]
* **CoPAL**: "使用大型语言模型纠正机器人行动规划", ICRA, 2023年10月. [[论文](https://arxiv.org/abs/2310.07263)] [[网站](https://hri-eu.github.io/Loom/)][[代码](https://github.com/HRI-EU/Loom/tree/main)]
* **LGMCTS**: "LGMCTS: 用于执行语义对象重排的语言引导蒙特卡洛树搜索", *arXiv, 2023年9月*. [[论文](https://arxiv.org/abs/2309.15821)]
* **Prompt2Walk**: "使用大型语言模型提示机器人行走", *arXiv, 2023年9月*, [[论文](https://arxiv.org/abs/2309.09969)] [[网站](https://prompt2walk.github.io)]
* **DoReMi**: "通过检测和恢复计划执行错位来构建语言模型", *arXiv, 2023年7月*, [[论文](https://arxiv.org/abs/2307.00329)] [[网站](https://sites.google.com/view/doremi-paper)]
* **Co-LLM-Agents**: "使用大型语言模型模块化构建合作具身代理", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2307.02485)] [[代码](https://github.com/UMass-Foundation-Model/Co-LLM-Agents)] [[网站](https://vis-www.cs.umass.edu/Co-LLM-Agents/)]
* **LLM-Reward**: "用于机器人技能合成的语言到奖励", *arXiv, 2023年6月*. [[论文](https://arxiv.org/abs/2306.08647)] [[网站](https://language-to-reward.github.io/)]
* **LLM-BRAIn**: "LLM-BRAIn: 基于大型语言模型的快速生成机器人行为树的AI驱动", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.19352)]
* **GLAM**: "通过在线强化学习在交互环境中接地大型语言模型", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2302.02662)] [[PyTorch代码](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)] 
* **LLM-MCTS**: "将大型语言模型作为常识知识用于大规模任务规划", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.14078v1)] 
* **AlphaBlock**: "AlphaBlock: 用于机器人操作中视觉语言推理的具身微调", arxiv, 2023年5月. [[论文](https://arxiv.org/abs/2305.18898)]
* **LLM+P**: "LLM+P: 赋予大型语言模型最优规划能力", *arXiv, 2023年4月*, [[论文](https://arxiv.org/abs/2304.11477)] [[代码](https://github.com/Cranial-XIX/llm-pddl)]
* **ChatGPT-Prompts**: "ChatGPT赋能各种环境中的长步机器人控制：案例应用", *arXiv, 2023年4月*, [[论文](https://arxiv.org/abs/2304.03893?s=03)] [[代码/提示](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts)]
* **ReAct**: "ReAct: 在语言模型中协同推理与行动", ICLR, 2023年4月. [[论文](https://arxiv.org/abs/2210.03629)] [[GitHub](https://github.com/ysymyth/ReAct)] [[网站](https://react-lm.github.io/)]
* **LLM-Brain**: "LLM作为机器人大脑：统一以自我为中心的记忆与控制", arXiv, 2023年4月. [[论文](https://arxiv.org/abs/2304.09349v1)] 
* "用于决策的基础模型：问题、方法和机会", *arXiv, 2023年3月*, [[论文](https://arxiv.org/abs/2303.04129)]
* **LLM-planner**: "LLM-Planner: 使用大型语言模型进行少样本具身代理规划", ICCV, 2023年3月. [[论文](https://arxiv.org/abs/2212.04088)] [[PyTorch代码](https://github.com/OSU-NLP-Group/LLM-Planner/)] [[网站](https://dki-lab.github.io/LLM-Planner/)]
* **Text2Motion**: "从自然语言指令到可行计划", *arXiV, 2023年3月*, [[论文](https://arxiv.org/abs/2303.12153)] [[网站](https://sites.google.com/stanford.edu/text2motion)]
* **GD**: "接地解码：用接地模型指导机器人控制的文本生成", *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.00855)] [[网站](https://grounded-decoding.github.io/)]
* **PromptCraft**: "ChatGPT用于机器人技术：设计原则和模型能力", 博客, 2023年2月, [[论文](https://arxiv.org/abs/2306.17582)] [[网站](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)]
* "使用语言模型设计奖励", *ICML, 2023年2月*. [[论文](https://arxiv.org/abs/2303.00001v1)] [[PyTorch代码](https://github.com/minaek/reward_design_with_llms)] 
* "通过修正再提示进行大型语言模型规划", *arXiv, 2022年11月*. [[论文](https://arxiv.org/abs/2311.09935)]
* **Don't Copy the Teacher**: "不要复制老师：具身对话中的数据和模型挑战", EMNLP, 2022年10月. [[论文](https://arxiv.org/abs/2210.04443)] [[网站](https://www.youtube.com/watch?v=qGPC65BDJw4&t=2s)]
* **COWP**: "开放世界中的机器人任务规划和情境处理", *arXiv, 2022年10月*. [[论文](https://arxiv.org/abs/2210.01287)] [[PyTorch代码](https://github.com/yding25/GPT-Planner)] [[网站](https://cowplanning.github.io/)]
* **LM-Nav**: "使用大型预训练的语言、视觉和动作模型进行机器人导航", *arXiv, 2022年7月*. [[论文](https://arxiv.org/abs/2207.04429)] [[PyTorch代码](https://github.com/blazejosinski/lm_nav)] [[网站](https://sites.google.com/view/lmnav)]
* **InnerMonologue**: "内心独白：通过语言模型规划进行具身推理", *arXiv, 2022年7月*. [[论文](https://arxiv.org/abs/2207.05608)] [[网站](https://innermonologue.github.io/)]
* **Housekeep**: "Housekeep: 使用常识推理整理虚拟家庭", *arXiv, 2022年5月*. [[论文](https://arxiv.org/abs/2205.10712)] [[PyTorch代码](https://github.com/yashkant/housekeep)] [[网站](https://yashkant.github.io/housekeep/)]
* **FILM**: "FILM: 使用模块化方法遵循语言指令", ICLR, 2022年4月. [[论文](https://arxiv.org/abs/2110.07342)] [[代码](https://github.com/soyeonm/FILM)] [[网站](https://soyeonm.github.io/FILM_webpage/)]
* **MOO**: "使用预训练视觉语言模型进行开放世界物体操作", *arXiv, 2022年3月*. [[论文](https://arxiv.org/abs/2303.00905)] [[网站](https://robot-moo.github.io/)]
* **LID**: "用于交互决策的预训练语言模型", *arXiv, 2022年2月*. [[论文](https://arxiv.org/abs/2202.01771)] [[PyTorch代码](https://github.com/ShuangLI59/Language-Model-Pre-training-Improves-Generalization-in-Policy-Learning)] [[网站](https://shuangli-project.github.io/Pre-Trained-Language-Models-for-Interactive-Decision-Making/)]
* "与语言模型协作进行具身推理", NeurIPS, 2022年2月. [[论文](https://arxiv.org/abs/2302.00763v1)]
* **ZSP**: "语言模型作为零样本规划器：提取具身代理的可操作知识", ICML, 2022年1月. [[论文](https://arxiv.org/abs/2201.07207)] [[PyTorch代码](https://github.com/huangwl18/language-planner)] [[网站](https://wenlong.page/language-planner/)]
* **CALM**: "保持冷静并探索：用于基于文本游戏的动作生成的语言模型", *arXiv, 2020年10月*. [[论文](https://arxiv.org/abs/2010.02903)] [[PyTorch代码](https://github.com/princeton-nlp/calm-textgame)] 
* "无需视觉的视觉规划：语言模型从高级指令推断详细计划", *arXiV, 2020年10月*, [[论文](https://arxiv.org/abs/2009.14259)] 

---

## 操作

* **Meta-Control**: "Meta-Control: 异构机器人技能的自动基于模型控制系统综合", CoRL, 2024年11月. [[论文](https://arxiv.org/abs/2405.11380)] [[网站](https://meta-control-paper.github.io/)]
* **A3VLM**: "A3VLM: 动作导向关节感知视觉语言模型", CoRL, 2024年11月. [[论文](https://arxiv.org/abs/2406.07549)] [[PyTorch代码](https://github.com/changhaonan/A3VLM)]
* **Manipulate-Anything**: "操纵一切：使用视觉语言模型自动化现实世界机器人", CoRL, 2024年11月. [[论文](https://arxiv.org/abs/2406.18915)] [[网站](https://robot-ma.github.io/)]
* **RobiButler**: "RobiButler: 家庭机器人助理的远程多模态交互", *arXiv, 2024年9月*. [[论文](https://arxiv.org/abs/2409.20548)] [[网站](https://robibutler.github.io/)]
* **SKT**: "SKT: 将状态感知关键点轨迹与视觉语言模型结合用于机器人衣物操作", *arXiv, 2024年9月*.  [[论文](https://arxiv.org/abs/2409.18082)] [[网站](https://sites.google.com/view/keypoint-garment/home)]
* **UniAff**: "UniAff: 使用视觉语言模型进行工具使用和关节操作的统一可供性表示", *arXiv, 2024年9月*.  [[论文](https://arxiv.org/abs/2409.20551)] [[网站](https://sites.google.com/view/uni-aff)]
* **Plan-Seq-Learn**: "Plan-Seq-Learn: 由语言模型引导的强化学习解决长时机器人任务", ICLR, 2024年5月. [[论文](https://arxiv.org/abs/2405.01534)], [[PyTorch代码](https://github.com/mihdalal/planseqlearn)] [[网站](https://mihdalal.github.io/planseqlearn/)]
* **ExploRLLM**: "ExploRLLM: 使用大型语言模型指导强化学习中的探索", *arXiv, 2024年3月*. [[论文](https://arxiv.org/abs/2403.09583)] [[网站](https://explorllm.github.io/)]
* **ManipVQA**: "ManipVQA: 将机器人可供性和物理基础信息注入多模态大型语言模型", IROS, 2024年3月, [[论文](https://arxiv.org/abs/2403.11289)] [[PyTorch代码](https://github.com/SiyuanHuang95/ManipVQA)] 
* **BOSS**: "Bootstrap Your Own Skills: 学习在LLM指导下解决新任务", CoRL, 2023年11月. [[论文](https://openreview.net/forum?id=a0mFRgadGO)] [[网站](https://clvrai.github.io/boss/)]
* **Lafite-RL**: "通过大型语言模型反馈加速机器人操作的强化学习", CoRL研讨会, 2023年11月. [[论文](https://arxiv.org/abs/2311.02379)]
* **Octopus**: "Octopus: 来自环境反馈的具身视觉语言程序员", *arXiv, 2023年10月*, [[论文](https://arxiv.org/abs/2310.08588)] [[PyTorch代码](https://github.com/dongyh20/Octopus)] [[网站](https://choiszt.github.io/Octopus/)]
* **[Text2Reward]** "Text2Reward: 自动密集奖励函数生成用于强化学习", *arXiv, 2023年9月*, [[论文](https://arxiv.org/abs/2309.11489)] [[网站](https://text-to-reward.github.io/)]
* **PhysObjects**: "物理基础视觉语言模型用于机器人操作", arxiv, 2023年9月. [[论文](https://arxiv.org/abs/2309.02561)]
* **[VoxPoser]** "VoxPoser: 使用语言模型进行机器人操作的可组合3D价值图", *arXiv, 2023年7月*, [[论文](https://arxiv.org/abs/2307.05973)] [[网站](https://voxposer.github.io/)]
* **Scalingup**: "扩展和蒸馏：语言指导的机器人技能获取", *arXiv, 2023年7月*, [[论文](https://arxiv.org/abs/2307.14535)] [[代码](https://github.com/columbia-ai-robotics/scalingup)] [[网站](https://www.cs.columbia.edu/~huy/scalingup/)]
 * **VoxPoser**: "VoxPoser: 使用语言模型进行机器人操作的可组合3D价值图", *arXiv, 2023年7月*. [[论文](https://arxiv.org/abs/2307.05973)] [[网站](https://voxposer.github.io/)]
 * **LIV**: "用于机器人控制的语言图像表示和奖励", *arXiv, 2023年6月*, [[论文](https://arxiv.org/abs/2306.00958)] [[PyTorch代码](https://github.com/penn-pal-lab/LIV)] [[网站](https://penn-pal-lab.github.io/LIV/)]
 * "语言指导的强化学习用于人机协调", *arXiv, 2023年6月*. [[论文](https://arxiv.org/abs/2304.07297)] 
* **RoboCat**: "RoboCat: 一个自我改进的机器人代理", arxiv, 2023年6月. [[论文](https://arxiv.org/abs/2306.11706)]  [[网站](https://www.deepmind.com/blog/robocat-a-self-improving-robotic-agent)]
* **SPRINT**: "通过语言指令重新标注进行语义策略预训练", arxiv, 2023年6月. [[论文](https://arxiv.org/abs/2306.11886)] [[网站](https://clvrai.github.io/sprint/)]
* **Grasp Anything**: "铺平道路以抓取任何东西：将基础模型转移到通用拾放机器人", arxiv, 2023年6月. [[论文](https://arxiv.org/abs/2306.05716)]
* **LLM-GROP**: "使用大型语言模型进行物体重排的任务和运动规划", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2303.06247)] [[网站](https://sites.google.com/view/llm-grop)]
* **VOYAGER**: "VOYAGER: 一个具有大型语言模型的开放式具身代理", *arXiv, 2023年5月*. [[论文](https://arxiv.org/abs/2305.16291)] [[PyTorch代码](https://github.com/MineDojo/Voyager)] [[网站](https://voyager.minedojo.org/)]
* **TIP**: "通过双图文提示进行多模态过程规划", *arXiV, 2023年5月*, [[论文](https://arxiv.org/abs/2305.01795)]
* **ProgramPort**: "程序基础、可组合泛化的机器人操作", ICLR, 2023年4月, [[论文](https://arxiv.org/abs/2304.13826)] [[网站] (https://progport.github.io/)]
* **VLaMP**: "预训练语言模型作为辅助人类的视觉规划器", *arXiV, 2023年4月*, [[论文](https://arxiv.org/abs/2304.09179)]
* "迈向具有基础模型的统一代理", ICLR, 2023年4月. [[论文](https://www.semanticscholar.org/paper/TOWARDS-A-UNIFIED-AGENT-WITH-FOUNDATION-MODELS-Palo-Byravan/67188a50e1d8a601896f1217451b99f646af4ac8)] 
* **CoTPC**: "思维链预测控制", *arXiv, 2023年4月*, [[论文](https://arxiv.org/abs/2304.00776)] [[代码](https://github.com/SeanJia/CoTPC)]
* **Plan4MC**: "Plan4MC: 开放世界《我的世界》任务的技能强化学习与规划", *arXiv, 2023年3月*. [[论文](https://arxiv.org/abs/2303.16563)] [[PyTorch代码](https://github.com/PKU-RL/Plan4MC)] [[网站](https://sites.google.com/view/plan4mc)]
* **ELL**: "使用大型语言模型指导强化学习中的预训练", *arXiv, 2023年2月*. [[论文](https://arxiv.org/abs/2302.06692)] 
* **DEPS**: "描述、解释、规划和选择：与大型语言模型的互动规划实现开放世界多任务代理", *arXiv, 2023年2月*. [[论文](https://arxiv.org/abs/2302.01560)] [[PyTorch代码](https://github.com/CraftJarvis/MC-Planner)]
* **LILAC**: "不，是右边 – 通过共享自主性的在线语言纠正进行机器人操作", *arXiv, 2023年1月*, [[论文](https://arxiv.org/abs/2301.02555)] [[PyTorch代码](https://github.com/Stanford-ILIAD/lilac)]
* **DIAL**: "通过指令增强与视觉语言模型进行机器人技能获取", *arXiv, 2022年11月*, [[论文](https://arxiv.org/abs/2211.11736)] [[网站](https://instructionaugmentation.github.io/)]
* **Gato**: "通用代理", *TMLR, 2022年11月*. [[论文](https://arxiv.org/abs/2205.06175)]  [[网站](https://www.deepmind.com/publications/a-generalist-agent)]
* **NLMap**: "现实世界规划的开放词汇查询场景表示", *arXiv, 2022年9月*, [[论文](https://arxiv.org/abs/2209.09874)] [[网站](https://nlmap-saycan.github.io/)]
* **R3M**: "机器人操作的通用视觉表示", *arXiv, 2022年11月*, [[论文](https://arxiv.org/abs/2203.12601)] [[PyTorch代码](https://github.com/facebookresearch/r3m)] [[网站](https://tinyurl.com/robotr3m)]
* **CLIP-Fields**: "弱监督机器人记忆语义场", *arXiv, 2022年10月*, [[论文](https://arxiv.org/abs/2210.05663)] [[PyTorch代码](https://github.com/notmahi/clip-fields)] [[网站](https://mahis.life/clip-fields/)]
* **VIMA**: "VIMA: 使用多模态提示的一般机器人操作", *arXiv, 2022年10月*, [[论文](https://arxiv.org/abs/2210.03094)] [[PyTorch代码](https://github.com/vimalabs/VIMA)] [[网站](https://vimalabs.github.io/)]
* **Perceiver-Actor**: "多任务机器人操作变压器", CoRL, 2022年9月. [[论文](https://arxiv.org/abs/2209.05451)] [[PyTorch代码](https://github.com/peract/peract)] [[网站](https://peract.github.io/)]
* **LaTTe**: "LaTTe: 语言轨迹变换器", *arXiv, 2022年8月*. [[论文](https://arxiv.org/abs/2208.02918)] [[TensorFlow代码](https://github.com/arthurfenderbucker/NL_trajectory_reshaper)] [[网站](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/robot-language/)]
* **Robots Enact Malignant Stereotypes**: "机器人实施恶性刻板印象", FAccT, 2022年6月. [[论文](https://arxiv.org/abs/2207.11569)] [[PyTorch代码](https://github.com/ahundt/RobotsEnactMalignantStereotypes)] [[网站](https://sites.google.com/view/robots-enact-stereotypes/home)] [[华盛顿邮报](https://www.washingtonpost.com/technology/2022/07/16/racist-robots-ai/)] [[Wired](https://www.wired.com/story/how-to-stop-robots-becoming-racist/)] (代码访问需申请)
* **ATLA**: "利用语言加速工具操作学习", CoRL, 2022年6月. [[论文](https://arxiv.org/abs/2206.13074)]
* **ZeST**: "基础模型能否为机器人操作提供零样本任务规范？", L4DC, 2022年4月. [[论文](https://arxiv.org/abs/2204.11134)]
* **LSE-NGU**: "从语言抽象和预训练表示进行语义探索", *arXiv, 2022年4月*. [[论文](https://arxiv.org/abs/2204.05080)]
* **MetaMorph**: "METAMORPH: 使用变压器学习通用控制器", arxiv, 2022年3月. [[论文](https://arxiv.org/abs/2203.11931)]
* **Embodied-CLIP**: "简单但有效：用于具身AI的CLIP嵌入", CVPR, 2021年11月. [[论文](https://arxiv.org/abs/2111.09888)] [[PyTorch代码](https://github.com/allenai/embodied-clip)]
* **CLIPort**: "CLIPort: 用于机器人操作的“什么”和“哪里”路径", CoRL, 2021年9月. [[论文](https://arxiv.org/abs/2109.12098)] [[PyTorch代码](https://github.com/cliport/cliport)] [[网站](https://cliport.github.io/)）

---

## 指令与导航

* **GSON**: "GSON: 基于群体的社会导航框架与大型多模态模型", arxiv, 2024年9月 [[论文](https://arxiv.org/abs/2409.18084)]
* **NaVid**: "NaVid: 基于视频的VLM为视觉与语言导航规划下一步", arxiv, 2024年3月 [[论文](https://arxiv.org/abs/2402.15852)] [[网站](https://pku-epic.github.io/NaVid)]
* **OVSG**: "基于开放词汇3D场景图的上下文感知实体基础", CoRL, 2023年11月. [[论文](https://openreview.net/forum?id=cjEI5qXoT0)] [[代码](https://github.com/changhaonan/OVSG)] [[网站](https://ovsg-l.github.io/)]
* **VLMaps**: "用于机器人导航的视觉语言地图", arXiv, 2023年3月. [[论文](https://arxiv.org/abs/2210.05714)] [[PyTorch代码](https://github.com/vlmaps/vlmaps)] [[网站](https://vlmaps.github.io/)]
* "Interactive Language: 实时与机器人交谈", arXiv, 2022年10月 [[论文](https://arxiv.org/abs/2210.06407)] [[网站](https://interactive-language.github.io/)]
* **NLMap**: "用于现实世界规划的开放词汇查询场景表示", arXiv, 2022年9月, [[论文](https://arxiv.org/abs/2209.09874)] [[网站](https://nlmap-saycan.github.io/)]
* **ADAPT**: "ADAPT: 使用模态对齐动作提示进行视觉语言导航", CVPR, 2022年5月. [[论文](https://arxiv.org/abs/2205.15509)]
* "预训练视觉模型在控制中的意外有效性", ICML, 2022年3月. [[论文](https://arxiv.org/abs/2203.03580)] [[PyTorch代码](https://github.com/sparisi/pvr_habitat)] [[网站](https://sites.google.com/view/pvr-control)]
* **CoW**: "CLIP on Wheels: 将零样本对象导航视为对象定位和探索", arXiv, 2022年3月. [[论文](https://arxiv.org/abs/2203.10421)]
* **Recurrent VLN-BERT**: "用于导航的循环视觉语言BERT", CVPR, 2021年6月 [[论文](https://arxiv.org/abs/2011.13922)] [[PyTorch代码](https://github.com/YicongHong/Recurrent-VLN-BERT)]
* **VLN-BERT**: "通过网络图像-文本对改善视觉语言导航", ECCV, 2020年4月 [[论文](https://arxiv.org/abs/2004.14973)] [[PyTorch代码](https://github.com/arjunmajum/vln-bert)]

---

## 模拟框架

* **ManiSkill3**: "ManiSkill3: 用于通用具身AI的GPU并行机器人仿真与渲染.", arxiv, 2024年10月. [[论文](https://arxiv.org/abs/2410.00425)] [[代码](https://github.com/haosulab/ManiSkill)] [[网站](http://maniskill.ai/)]
 * **GENESIS**: "用于通用机器人和具身AI学习的生成世界.", arXiv, 2023年11月. [[代码](https://github.com/Genesis-Embodied-AI/Genesis)] 
 * **ARNOLD**: "ARNOLD: 一个连续状态的真实3D场景中语言基础任务学习基准", ICCV, 2023年4月. [[论文](https://arxiv.org/abs/2304.04321)] [[代码](https://github.com/arnold-benchmark/arnold)] [[网站](https://arnold-benchmark.github.io/)]
 * **OmniGibson**: "基于NVIDIA Omniverse引擎加速具身AI研究的平台". 第六届机器人学习年度会议, 2022年. [[论文](https://openreview.net/forum?id=_8DoIe8G3t)] [[代码](https://github.com/StanfordVL/OmniGibson)] 
 * **MineDojo**: "MineDojo: 构建具有互联网规模知识的开放式具身代理", arXiv, 2022年6月. [[论文](https://arxiv.org/abs/2206.08853)] [[代码](https://github.com/MineDojo/MineDojo)] [[网站](https://minedojo.org/)] [[开放数据库](https://minedojo.org/knowledge_base.html)]
 * **Habitat 2.0**: "Habitat 2.0: 训练家庭助手重组其居住地", NeurIPS, 2021年12月. [[论文](https://arxiv.org/abs/2106.14405)] [[代码](https://github.com/facebookresearch/habitat-sim)] [[网站](https://aihabitat.org/)]
 * **BEHAVIOR**: "BEHAVIOR: 虚拟、互动和生态环境中日常家庭活动的基准", CoRL, 2021年11月. [[论文](https://arxiv.org/abs/2108.03332)] [[代码](https://github.com/StanfordVL/behavior)] [[网站](https://behavior.stanford.edu/)]
 * **iGibson 1.0**: "iGibson 1.0: 大型真实场景中交互任务的仿真环境", IROS, 2021年9月. [[论文](https://arxiv.org/abs/2012.02924)] [[代码](https://github.com/StanfordVL/iGibson)] [[网站](https://svl.stanford.edu/igibson/)]
 * **ALFRED**: "ALFRED: 解释日常任务地面指令的基准", CVPR, 2020年6月. [[论文](https://arxiv.org/abs/1912.01734)] [[代码](https://github.com/askforalfred/alfred)] [[网站](https://askforalfred.com/)]
  * **BabyAI**: "BabyAI: 研究地面语言学习样本效率的平台", ICLR, 2019年5月. [[https://arxiv.org/abs/1810.08272)] [[代码](https://github.com/mila-iqia/babyai/tree/iclr19)]

---

## 安全、风险、红队测试及对抗性测试

* **RoboPAIR**: "破解LLM控制的机器人", 国际机器人与自动化会议 (ICRA) 2025年5月. [[论文](https://arxiv.org/abs/2410.13691)] [[网站](https://robopair.org/)]
* **RoboGuard**: "LLM启用机器人的安全护栏", arXiv, 2025年4月. [[论文](https://arxiv.org/abs/2503.07885)] [[网站](https://robo-guard.github.io/)]
* **Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis** arXiv, 2025年3月 [[arXiv](https://arxiv.org/abs/2503.03911)] [[代码](https://github.com/TUM-CPS-HN/SafeLLMRA)]
* **LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions** arXiv, 2024年6月. [[论文](https://arxiv.org/abs/2406.08824)]
* **Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics** arXiv, 2024年2月. [[论文](https://arxiv.org/abs/2402.10340)]
* **Robots Enact Malignant Stereotypes** FAccT, 2022年6月. [[arXiv](https://arxiv.org/abs/2207.11569)] [[DOI](https://doi.org/10.1145/3531146.3533138)] [[代码](https://github.com/ahundt/RobotsEnactMalignantStereotypes)] [[网站](https://sites.google.com/view/robots-enact-stereotypes/home)]
* **Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics** arXiv, 2024年11月 [[arXiv](https://arxiv.org/abs/2411.13587)] [[代码](https://github.com/William-wAng618/roboticAttack)] [[网站](https://vlaattacker.github.io/)]

---

## 引用

如果您觉得此仓库有用，请考虑引用：

```
@misc{kira2022llmroboticspaperslist,
    title = {Awesome-LLM-Robotics},
    author = {Zsolt Kira},
    journal = {GitHub repository},
    url = {https://github.com/GT-RIPL/Awesome-LLM-Robotics},
    year = {2022},
}
```
