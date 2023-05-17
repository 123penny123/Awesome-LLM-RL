# **Awesome-LM-RL** #
By Guangran Cheng.

2023 up-to-date list of PAPERS, CODEBASES, and BENCHMARKS on Decision Making using Foundation Models including LLMs and VLMs.

Please feel free to send me pull requests or contact me to correct any mistakes.

---
## **Table of Contents** ##

- [Survey of Foundation Models in Decision Making](#Survey)
- [Foundation Models as World Models](#World-Models)
- [Foundation Models as Reward Models](#Reward-Models)
- [Foundation Models as Planners](#Planners)
- [Foundation Models as Representation Encoders](#Encoders)
- [Multi-modal Decision Making Benchmarks](#Benchmark)

---

## **Paper** ##

### **Survey** ###
- "A survey of reinforcement learning informed by natural language." arXiv, 2019. [[paper]](https://arxiv.org/pdf/1906.03926)
- "A Survey on Transformers in Reinforcement Learning." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2301.03044)
- "Foundation models for decision making: Problems, methods, and opportunities." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2303.04129)

### **World Models** ###
- **IRIS**: "Transformers are sample efficient world models." ICLR, 2023. [[paper]](https://arxiv.org/pdf/2209.00588)[[code]](https://github.com/eloialonso/iris)
- **UniPi**: "Learning Universal Policies via Text-Guided Video Generation." arXiv, 2023.[[paper]](https://arxiv.org/pdf/2302.00111)[[website]](https://universal-policy.github.io/unipi/)

### **Reward Models** ###
- "Reward design with language models." ICLR, 2023. [[paper]](https://arxiv.org/pdf/2303.00001)[[code]](https://github.com/minaek/reward_design_with_llms)
- **ELLM**: "Guiding Pretraining in Reinforcement Learning with Large Language Models." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2302.06692)
### **Planners** ###
- **FILM**: "Film: Following instructions in language with modular methods." ICLR, 2022. [[paper]](https://arxiv.org/pdf/2110.07342)[[code]](https://soyeonm.github.io/FILM_webpage/)[[website]](https://soyeonm.github.io/FILM_webpage/)
- **SayCan**: "Do as i can, not as i say: Grounding language in robotic affordances." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2204.01691)[[code]](https://github.com/google-research/google-research/tree/master/saycan)[[website]](https://say-can.github.io/)
- **PaLM-E**: "Palm-e: An embodied multimodal language model." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2303.03378.pdf?trk=public_post_comment-text)[[website]](https://palm-e.github.io/)
- **LM-Nav**: "Lm-nav: Robotic navigation with large pre-trained models of language, vision, and action." CoRL, 2022.[[paper]](https://proceedings.mlr.press/v205/shah23b/shah23b.pdf)[[code]](https://github.com/blazejosinski/lm_nav)[[website]](https://sites.google.com/view/lmnav)
- **ZSP**: "Language models as zero-shot planners: Extracting actionable knowledge for embodied agents." ICML, 2022. [[paper]](https://proceedings.mlr.press/v162/huang22a/huang22a.pdf)[[code]](https://github.com/huangwl18/language-planner)[[website]](https://wenlong.page/language-planner/)
- "Grounding large language models in interactive environments with online reinforcement learning." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2302.02662)[[code]](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)
- **DEPS**: "Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2302.01560)[[code]](https://github.com/CraftJarvis/MC-Planner)
- **Inner Monologue**: "Inner monologue: Embodied reasoning through planning with language models." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2207.05608)[[website]](https://innermonologue.github.io/)
- **Plan4MC**: "Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2303.16563)[[code]](https://github.com/PKU-RL/Plan4MC)[[website]](https://sites.google.com/view/plan4mc)
- **TidyBot**: "TidyBot: Personalized Robot Assistance with Large Language Models." arXiv, 2023. [[paper]](https://arxiv.org/pdf/2305.05658)[[website]](https://tidybot.cs.princeton.edu/)

### **Encoders** ###
- **Cliport**: "Cliport: What and where pathways for robotic manipulation." CoRL, 2021. [[paper]](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf)[[code]](https://github.com/cliport/cliport)[[website]](https://cliport.github.io/)
- **Vima**; "Vima: General robot manipulation with multimodal prompts." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2210.03094)[[code]](https://github.com/vimalabs/VIMA)[[website]](https://vimalabs.github.io/)
- **Perceiver-actor**: "Perceiver-actor: A multi-task transformer for robotic manipulation." CoRL, 2022. [[paper]](https://proceedings.mlr.press/v205/shridhar23a/shridhar23a.pdf)[[code]](https://github.com/peract/peract)[[website]](https://peract.github.io/)
- **InstructRL**: "Instruction-Following Agents with Jointly Pre-Trained Vision-Language Models." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2210.13431)
- **Hiveformer**: "Instruction-driven history-aware policies for robotic manipulations." CoRL, 2022. [[paper]](https://proceedings.mlr.press/v205/guhur23a/guhur23a.pdf)[[code]](https://github.com/guhur/hiveformer)[[website]](https://guhur.github.io/hiveformer/)
- **LID**: "Pre-trained language models for interactive decision-making." NIPS, 2022. [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/ca3b1f24fc0238edf5ed1ad226b9d655-Paper-Conference.pdf)[[code]](https://github.com/ShuangLI59/Pre-Trained-Language-Models-for-Interactive-Decision-Making)[[website]](https://shuangli-project.github.io/Pre-Trained-Language-Models-for-Interactive-Decision-Making/)
- **LISA**: "LISA: Learning Interpretable Skill Abstractions from Language." NIPS, 2022. [[paper]](https://arxiv.org/pdf/2203.00054)[[code]](https://github.com/Div99/LISA)
- **LoReL**: "Learning language-conditioned robot behavior from offline data and crowd-sourced annotation." CoRL, 2021. [[paper]](https://proceedings.mlr.press/v164/nair22a/nair22a.pdf)[[code]](https://github.com/suraj-nair-1/lorel)[[website]](https://sites.google.com/view/robotlorel)

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

### **Game** ###
- **Minedojo** "Minedojo: Building open-ended embodied agents with internet-scale knowledge." arXiv, 2022. [[paper]](https://arxiv.org/pdf/2206.08853)[[code]](https://github.com/MineDojo/MineDojo)[[website]](https://minedojo.org/)
- **BabyAI**  "Babyai: A platform to study the sample efficiency of grounded language learning." ICLR, 2019. [[paper]](https://arxiv.org/pdf/1810.08272)[[code]](https://github.com/mila-iqia/babyai)

---
## **Citation** ##

