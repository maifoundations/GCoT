<h1 align="center">Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation</h1>
<p align="center"><i>A bootstrapping, annotation-free method to inject grounding information into Chain-of-Thought, enabling data-efficient adaptation.</i></p>

<p align="center">
          📑 <a href="https://arxiv.org/pdf/2507.02859">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://www.maifoundations.com/blog/gcot/">Blog</a> &nbsp&nbsp 
</p>

This is the official implementation of the paper 'Bootstrapping Grounded Chain-of-Thought in Multimodal LLMs for Data-Efficient Model Adaptation'.

# News📰
* **`[2025/07/03]`:**🔥**We have released our paper [[Arxiv](https://arxiv.org/pdf/2507.02859)].**
* **`[2025/06/26]`:**🎉**GCoT is accepted by ICCV 2025**

# Overview✈️
To ensure that MLLMs can adapt and excel in specialized applications, we propose the Grounded Chain-of-Thought (GCoT) approach. This simple yet effective strategy aims to inject grounding information into CoT data, enhancing the fidelity of reasoning steps to input images. By doing so, models trained with GCoT data can potentially achieve better generalization with limited training samples. Given the challenges in collecting grounded CoT data, we introduce a straightforward bootstrapping method: iteratively using an MLLM to generate grounding labels and refining them through self-verification.

<video src="/assets/demo.mp4" width="720" controls></video>

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

