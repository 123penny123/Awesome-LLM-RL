# **Awesome-LLM-Decision-Making** #

2023 up-to-date list of PAPERS, CODEBASES, and BENCHMARKS on Decision Making using Foundation Models including LLMs and VLMs.

Please feel free to send me pull requests or contact me to correct any mistakes.

---
## **Table of Contents** ##

- [Survey of Foundation Models in Decision Making](#Survey)
- [Foundation Models as World Models](#World-Models)
- [Foundation Models as Reward Models](#Reward-Models)
- [Foundation Models as Agent Models](#Agent-Models)
- [Foundation Models as Representation Encoders](#Encoders)
- [Multi-modal Decision Making Benchmarks](#Benchmark)

---

## **Paper** ##

### **Survey** ###
- "A survey of reinforcement learning informed by natural language." arXiv, 2019. [[paper]](https://arxiv.org/pdf/1906.03926)
- "A Survey on Transformers in Reinforcement Learning." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2301.03044)
- "Foundation models for decision making: Problems, methods, and opportunities." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2303.04129)
- "A Survey of Large Language Models." arXiv, June 2023. [[paper]](https://arxiv.org/pdf/2303.18223)[[code]](https://github.com/RUCAIBox/LLMSurvey)
- "A Survey on Large Language Model based Autonomous Agents." arXiv, Aug 2023. [[paper]](https://arxiv.org/pdf/2308.11432)[[code]](https://github.com/Paitesanshi/LLM-Agent-Survey)
- "Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security." arXiv, Jan 2024. [[paper]](https://arxiv.org/pdf/2401.05459.pdf)[[code]](https://github.com/MobileLLM/Personal_LLM_Agents_Survey)

### **World Models** ###
- **IRIS**: "Transformers are sample efficient world models." ICLR, 2023. [[paper]](https://arxiv.org/pdf/2209.00588)[[code]](https://github.com/eloialonso/iris)
- **UniPi**: "Learning Universal Policies via Text-Guided Video Generation." arXiv, 2023.[[paper]](https://arxiv.org/pdf/2302.00111)[[website]](https://universal-policy.github.io/unipi/)
- **Dynalang**： "Learning to Model the World with Language." arXiv, July 2023. [[paper]](https://arxiv.org/pdf/2308.01399)[[website]](https://dynalang.github.io/)[[code]](https://github.com/jlin816/dynalang)

### **Reward Models** ###
- **EAGER**: "EAGER: Asking and Answering Questions for Automatic Reward Shaping in Language-guided RL." NIPS, 2022. [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/50eb39ab717507cccbe2b8590de32030-Paper-Conference.pdf)[[code]](https://github.com/flowersteam/eager)
- "Reward design with language models." ICLR, 2023. [[paper]](https://arxiv.org/pdf/2303.00001)[[code]](https://github.com/minaek/reward_design_with_llms)
- **ELLM**: "Guiding Pretraining in Reinforcement Learning with Large Language Models." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2302.06692)
- "Language to Rewards for Robotic Skill Synthesis." arXiv, June 2023. [[paper]](https://arxiv.org/pdf/2306.08647)[[website]](https://language-to-reward.github.io/)
- "Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning." arXiv, Oct 2023. [[paper]](https://arxiv.org/pdf/2310.12921.pdf)
- **Eureka**: "Eureka: Human-Level Reward Design via Coding Large Language Models." arXiv, Oct 2023. [[paper]](https://arxiv.org/pdf/2310.12931.pdf)[[wensite]](https://eureka-research.github.io/)[[code]](https://github.com/eureka-research/Eureka)

### **Agent Models** ###
- **Generative Agent**
  - **FILM**: "Film: Following instructions in language with modular methods." ICLR, 2022. [[paper]](https://arxiv.org/pdf/2110.07342)[[code]](https://soyeonm.github.io/FILM_webpage/)[[website]](https://soyeonm.github.io/FILM_webpage/)
  - "Grounding large language models in interactive environments with online reinforcement learning." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2302.02662)[[code]](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)
  - **Inner Monologue**: "Inner monologue: Embodied reasoning through planning with language models." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2207.05608)[[website]](https://innermonologue.github.io/)
  - **Plan4MC**: "Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2303.16563)[[code]](https://github.com/PKU-RL/Plan4MC)[[website]](https://sites.google.com/view/plan4mc)
  - **ProgPrompt**: "ProgPrompt: Generating Situated Robot Task Plans using Large Language Models." ICRA, 2023. [[paper]](https://arxiv.org/pdf/2209.11302)[[website]](https://progprompt.github.io/)
  - **Text2Motion**: "Text2Motion: From Natural Language Instructions to Feasible Plans." arXiv, Mar 2023. [[paper]](https://arxiv.org/pdf/2303.12153)[[website]](https://sites.google.com/stanford.edu/text2motion)
  - **Voyager**: "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv, May 2023. [[paper]](https://arxiv.org/pdf/2305.16291)[[code]](https://github.com/MineDojo/Voyager)[[website]](https://voyager.minedojo.org/)
  - **Reflexion**: "Reflexion: Language Agents with Verbal Reinforcement Learning." arXiv, Mar 2023. [[paper]](https://arxiv.org/pdf/2303.11366)[[code]](https://github.com/noahshinn024/reflexion)
  - **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR, 2023. [[paper]](https://arxiv.org/pdf/2210.03629)[[code]](https://github.com/ysymyth/ReAct)[[website]](https://react-lm.github.io/)
  - "Generative Agents: Interactive Simulacra of Human Behavior." arXiv, Apr 2023. [[paper]](https://arxiv.org/pdf/2304.03442.pdf%C3%82%C2%A0)[[code]](https://github.com/joonspk-research/generative_agents)
  - "Cognitive Architectures for Language Agents." arXiv, Sep 2023. [[paper]](https://arxiv.org/abs/2309.02427)[[code]](https://github.com/ysymyth/awesome-language-agents)
  - **Retroformer**: "Retroformer: Retrospective large language agents with policy gradient optimization." arXiv, Aug 2023. [[paper]](https://arxiv.org/pdf/2308.02151.pdf)
  - **SayCanPay**: "Heuristic Planning with Large Language Models using Learnable Domain Knowledge." AAAI, 2024. [[paper]](https://arxiv.org/pdf/2308.12682.pdf)[[code]](https://github.com/RishiHazra/saycanpay)[[website]](https://rishihazra.github.io/SayCanPay/)
  - **SPAG**: "Self-playing Adversarial Language Game Enhances LLM Reasoning." arXiv, 2024. [[paper]](https://arxiv.org/abs/2404.10642)[[code]](https://github.com/Linear95/SPAG)
  
- **Embodied AI**
  - **SayCan**: "Do as i can, not as i say: Grounding language in robotic affordances." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2204.01691)[[code]](https://github.com/google-research/google-research/tree/master/saycan)[[website]](https://say-can.github.io/)
  - **PaLM-E**: "Palm-e: An embodied multimodal language model." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2303.03378.pdf?trk=public_post_comment-text)[[website]](https://palm-e.github.io/)
  - **LM-Nav**: "Lm-nav: Robotic navigation with large pre-trained models of language, vision, and action." CoRL, 2022.[[paper]](https://proceedings.mlr.press/v205/shah23b/shah23b.pdf)[[code]](https://github.com/blazejosinski/lm_nav)[[website]](https://sites.google.com/view/lmnav)
  - **ZSP**: "Language models as zero-shot planners: Extracting actionable knowledge for embodied agents." ICML, 2022. [[paper]](https://proceedings.mlr.press/v162/huang22a/huang22a.pdf)[[code]](https://github.com/huangwl18/language-planner)[[website]](https://wenlong.page/language-planner/)
  - **DEPS**: "Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2302.01560)[[code]](https://github.com/CraftJarvis/MC-Planner)
  - **TidyBot**: "TidyBot: Personalized Robot Assistance with Large Language Models." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2305.05658)[[website]](https://tidybot.cs.princeton.edu/)
  - **Chatgpt for robotics**: "Chatgpt for robotics: Design principles and model abilities." Microsoft Auton. Syst. Robot. Res 2 (2023): 20. [[paper]](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf)
  - **KNOWNO**: "Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners." arXiv, July 2023. [[paper]](https://arxiv.org/pdf/2307.01928)
  - **VoxPoser**: "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models." July 2023. [[[paper]](https://voxposer.github.io/voxposer.pdf)[[website]](https://voxposer.github.io/)
  - **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale." arXiv, Dec 2022. [[paper]](https://arxiv.org/pdf/2212.06817)[[code]](https://github.com/google-research/robotics_transformer)
  - **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." Deepmind, July 2023. [[paper]](https://robotics-transformer2.github.io/assets/rt2.pdf)[[website]](https://robotics-transformer2.github.io/)
  - **MOO**: "Open-World Object Manipulation using Pre-trained Vision-Language Models." arXiv, Mar 2023. [[paper]](https://arxiv.org/pdf/2303.00905)[[website]](https://robot-moo.github.io/)
  - **EmbodiedGPT**: "EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought." arXiv, May 2023. [[paper]](https://arxiv.org/pdf/2305.15021)[[code]](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch)[[website]](https://embodiedgpt.github.io/)
  - **RoboCat**: "RoboCat: A self-improving robotic agent." arXiv, Jun 2023. [[paper]](https://arxiv.org/abs/2306.11706)[[website]](https://www.deepmind.com/blog/robocat-a-self-improving-robotic-agent)
  - **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." [[paper]](https://robotics-transformer-x.github.io/paper.pdf)[[code]](https://console.cloud.google.com/storage/browser/gresearch/robotics/open_x_embodiment_and_rt_x_oss;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)[[website]](https://robotics-transformer-x.github.io/)
  - **GenSim**: "GenSim: Generating Robotic Simulation Tasks via Large Language Models." arXiv, Oct 2023. [[paper]](https://arxiv.org/abs/2310.01361)[[code]](https://github.com/liruiw/GenSim)[[website]](https://liruiw.github.io/gensim/)
  - "Language Models as Zero-Shot Trajectory Generators." arXiv, Oct 2023. [[paper]](https://arxiv.org/abs/2310.11604)[[code]](https://github.com/kwonathan/language-models-trajectory-generators)[[website]](https://www.robot-learning.uk/language-models-trajectory-generators)
  - **LLaRP**: "Large Language Models as Generalizable Policies for Embodied Tasks." arXiv, Oct 2023. [[paper]](https://arxiv.org/pdf/2310.17722.pdf)[[website]](https://llm-rl.github.io/)
  - **CLARA**: "CLARA: Classifying and Disambiguating User Commands for Reliable Interactive Robotic Agents." arXiv, June 2023. [[paper]](https://arxiv.org/pdf/2306.10376.pdf)
  - **Ada**: "Learning adaptive planning representations with natural language guidance." arXiv, Dec 2023. [[paper]](https://arxiv.org/pdf/2312.08566.pdf)
  - "Demonstrating Large Language Models on Robots." RSS, 2023. [[paper]](https://www.roboticsproceedings.org/rss19/p024.pdf)
  - "Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents." NIPS, 2023. [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/bb3cfcb0284642a973dd631ec9184f2f-Paper-Conference.pdf)
  - "Learning to Learn Faster from Human Feedback with Language Model Predictive Control." arXiv, Feb 2024. [[paper]](https://arxiv.org/pdf/2402.11450)
 
### **Representation** ###
- **Cliport**: "Cliport: What and where pathways for robotic manipulation." CoRL, 2021. [[paper]](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf)[[code]](https://github.com/cliport/cliport)[[website]](https://cliport.github.io/)
- **Vima**; "Vima: General robot manipulation with multimodal prompts." ICML, 2023. [[paper]](https://arxiv.org/pdf/2210.03094)[[code]](https://github.com/vimalabs/VIMA)[[website]](https://vimalabs.github.io/)
- **Perceiver-actor**: "Perceiver-actor: A multi-task transformer for robotic manipulation." CoRL, 2022. [[paper]](https://proceedings.mlr.press/v205/shridhar23a/shridhar23a.pdf)[[code]](https://github.com/peract/peract)[[website]](https://peract.github.io/)
- **InstructRL**: "Instruction-Following Agents with Jointly Pre-Trained Vision-Language Models." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2210.13431)
- **Hiveformer**: "Instruction-driven history-aware policies for robotic manipulations." CoRL, 2022. [[paper]](https://proceedings.mlr.press/v205/guhur23a/guhur23a.pdf)[[code]](https://github.com/guhur/hiveformer)[[website]](https://guhur.github.io/hiveformer/)
- **LID**: "Pre-trained language models for interactive decision-making." NIPS, 2022. [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/ca3b1f24fc0238edf5ed1ad226b9d655-Paper-Conference.pdf)[[code]](https://github.com/ShuangLI59/Pre-Trained-Language-Models-for-Interactive-Decision-Making)[[website]](https://shuangli-project.github.io/Pre-Trained-Language-Models-for-Interactive-Decision-Making/)
- **LISA**: "LISA: Learning Interpretable Skill Abstractions from Language." NIPS, 2022. [[paper]](https://arxiv.org/pdf/2203.00054)[[code]](https://github.com/Div99/LISA)
- **LoReL**: "Learning language-conditioned robot behavior from offline data and crowd-sourced annotation." CoRL, 2021. [[paper]](https://proceedings.mlr.press/v164/nair22a/nair22a.pdf)[[code]](https://github.com/suraj-nair-1/lorel)[[website]](https://sites.google.com/view/robotlorel)
- **GRIF**: "Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2307.00117)[[website]](https://rail-berkeley.github.io/grif/)

---
## **Benchmark** ##

### **Manipulation** ###
- **Meta-World**: "Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning." CoRl, 2019. [[paper]](http://proceedings.mlr.press/v100/yu20a/yu20a.pdf)[[code]](https://github.com/Farama-Foundation/Metaworld)[[website]](https://meta-world.github.io/)
- **RLbench**: James, Stephen, et al. "Rlbench: The robot learning benchmark & learning environment." IEEE Robotics and Automation Letters, 2020. [[paper]](https://arxiv.org/pdf/1909.12271)[[code]](https://github.com/stepjam/RLBench)[[website]](https://sites.google.com/view/rlbench)
- **VLMbench**: Zheng, Kaizhi, et al. "Vlmbench: A compositional benchmark for vision-and-language manipulation." NIPS, 2022. [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/04543a88eae2683133c1acbef5a6bf77-Paper-Datasets_and_Benchmarks.pdf)[[code]](https://github.com/eric-ai-lab/vlmbench)[[website]](https://sites.google.com/ucsc.edu/vlmbench/home)
- **Calvin**: Mees, Oier, et al. "Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks." IEEE Robotics and Automation Letters, 2022. [[paper]](https://arxiv.org/pdf/2112.03227)[[code]](https://github.com/mees/calvin)[[website]](http://calvin.cs.uni-freiburg.de/)

### **Navigation-and-Manipulation** ###
- **AI2-THOR** "Ai2-thor: An interactive 3d environment for visual ai." arXiv, 2017. [[paper]](https://arxiv.org/pdf/1712.05474)[[code]](https://github.com/allenai/ai2thor)[[website]](https://ai2thor.allenai.org/)
- **Alfred**: "Alfred: A benchmark for interpreting grounded instructions for everyday tasks." CVPR, 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.pdf)[[code]](https://github.com/askforalfred/alfred)[[website]](https://askforalfred.com/)
- **VirtualHome**: "Watch-and-help: A challenge for social perception and human-ai collaboration." arXiv, 2020. [[paper]](https://arxiv.org/pdf/2010.09890)[[code]](https://github.com/xavierpuigf/watch_and_help)[[website]](http://virtual-home.org/watch_and_help/)
- **Ravens**: "Transporter networks: Rearranging the visual world for robotic manipulation." CoRL, 2020. [[paper]](https://arxiv.org/pdf/2010.14406.pdf)[[code]](https://github.com/google-research/ravens)[[website]](https://transporternets.github.io/)
- **Housekeep**: "Housekeep: Tidying virtual households using commonsense reasoning." ECCV, 2022. [[paper]](https://arxiv.org/pdf/2205.10712)[[code]](https://github.com/yashkant/housekeep)[[website]](https://yashkant.github.io/housekeep/#:~:text=Abstract,objects%20need%20to%20be%20rearranged.)
- **Behavior-1k**: "Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation." CoRL, 2022. [[paper]](https://proceedings.mlr.press/v205/li23a/li23a.pdf)[[code]](https://github.com/StanfordVL/OmniGibson)[[website]](https://behavior.stanford.edu/behavior-1k)
- **Habitat 2.0**: "Habitat 2.0: Training home assistants to rearrange their habitat." NIPS, 2021. [[paper]](https://proceedings.neurips.cc/paper/2021/file/021bbc7ee20b71134d53e20206bd6feb-Paper.pdf)[[code]](https://github.com/facebookresearch/habitat-lab)[[website]](https://aihabitat.org/docs/habitat2/)
- **EgoTV** 📺: "Egocentric Task Verification from Natural Language Task Descriptions." ICCV, 2023 (from Meta AI) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Hazra_EgoTV_Egocentric_Task_Verification_from_Natural_Language_Task_Descriptions_ICCV_2023_paper.pdf)[[code]](https://github.com/facebookresearch/EgoTV)[[website]](https://rishihazra.github.io/EgoTV)

### **Game** ###
- **Minedojo**: "Minedojo: Building open-ended embodied agents with internet-scale knowledge." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2206.08853)[[code]](https://github.com/MineDojo/MineDojo)[[website]](https://minedojo.org/)
- **BabyAI**:  "Babyai: A platform to study the sample efficiency of grounded language learning." ICLR, 2019. [[paper]](https://arxiv.org/pdf/1810.08272)[[code]](https://github.com/mila-iqia/babyai)
- **Generative Agents**: "Generative Agents: Interactive Simulacra of Human Behavior." arXiv Apr 2023. [[paper]](https://arxiv.org/pdf/2304.03442)[[website]](https://reverie.herokuapp.com/arXiv_Demo/#)[[code]](https://github.com/joonspk-research/generative_agents)
- **AgentBench**: "AgentBench: Evaluating LLMs as Agents." arXiv, Aug 2023. [[paper]](https://arxiv.org/pdf/2308.03688)[[website]](https://llmbench.ai/)[[code]](https://github.com/THUDM/AgentBench)

### **Tools** ###
- **Toolformer**: "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv, Feb 2023. [[paper]](https://arxiv.org/pdf/2302.04761)[[code]](https://github.com/lucidrains/toolformer-pytorch/tree/main)

---
## **Citation** ##

