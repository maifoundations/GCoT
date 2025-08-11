<h1 align="center">Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation</h1>
<p align="center"><i>An annotation-free method to inject grounding information into Chain-of-Thought, enabling data-efficient adaptation.</i></p>

<p align="center">
          📑 <a href="https://arxiv.org/pdf/2507.02859">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://www.maifoundations.com/blog/gcot/">Blog</a> &nbsp&nbsp 
</p>

This is the official implementation of the paper 'Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation'.

# News📰
* **`[2025/07/24]`:**🎉**GCoT is selected as ${\color{red}Highlight}$ Paper of ICCV 2025**
* **`[2025/07/03]`:**🔥**We have released our paper [[Arxiv](https://arxiv.org/pdf/2507.02859)].**
* **`[2025/06/26]`:**🎉**GCoT is accepted by ICCV 2025**

# Overview✈️
To ensure that MLLMs can adapt and excel in specialized applications, we propose the Grounded Chain-of-Thought (GCoT) approach. This simple yet effective strategy aims to inject grounding information into CoT data, enhancing the fidelity of reasoning steps to input images. By doing so, models trained with GCoT data can potentially achieve better generalization with limited training samples. Given the challenges in collecting grounded CoT data, we introduce a straightforward bootstrapping method: iteratively using an MLLM to generate grounding labels and refining them through self-verification.

<img src="assets/Bootstrapping.gif">

# Set up 📐
## Environment
```
git@github.com:maifoundations/GCoT.git
cd GCoT

# build environment
conda create -n GCoT python=3.9
conda activate GCoT

pip install -e .
```

## Data Preparation
To start the bootstrapping loop, the data should be structured in the following format. We added the "cot" data on top of the llava data format.
```
    {
        "id": 17449,
        "image": "29099.png",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nLook at the table. Then answer the question. At a price of $325, is there a shortage or a surplus?\nOptions:\nshortage\nsurplus"
            },
            {
                "from": "gpt",
                "value": "shortage"
            }
        ],
        "ques_type": "multi_choice",
        "cot": "{To determine whether there is a shortage or surplus at a price of $325, we need to compare the quantity demanded and the quantity supplied at that price.\\n\\nStep 1: Identify the quantity demanded at a price of $325. According to the table, at a price of $325, the quantity demanded is 10,600.\\n\\nStep 2: Identify the quantity supplied at a price of $325. According to the table, at a price of $325, the quantity supplied is 7,900.\\n\\nStep 3: Compare the quantity demanded and the quantity supplied. Since the quantity demanded (10,600) is greater than the quantity supplied (7,900), there is a shortage.\\n\\n*Answer*: shortage\"}",
    }
```

## Training
You can start the bootstrapping loop on the structured data using the following script:
```
sh bootstrapping.sh data.json PATH_TO_IMAGE
```

Additionally, you can modify the script to adjust the number of bootstrapping iterations and the training sample size.
With this script, you can easily equip your CoT with grounding information.

# Citation🎓
```
@article{xia2025bootstrapping,
  title={Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation},
  author={Xia, Jiaer and Tong, Bingkui and Zang, Yuhang and Shao, Rui and Zhou, Kaiyang},
  journal={arXiv preprint arXiv:2507.02859},
  year={2025}
}
```
# Acknowledgment
Our code is developed using the [LLaVA](https://github.com/haotian-liu/LLaVA) repository, and the experiments are conducted based on the [Visual-CoT](https://github.com/deepcs233/Visual-CoT) model.

