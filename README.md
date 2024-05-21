# Open-Genie: Generative Interactive Environments in PyTorch

This repo contains the _unofficial_ implementation of _Genie: Generative Interactive Environments_ [Bruce et al. (2024)](https://arxiv.org/abs/2402.15391v1?curius=4125) as introduced by Google DeepMind.

The goal of the model is to introduce "[...] The first generative interactive environment trained in an unsupervised manner from unlabelled Internet videos".

![Genie Banner](res/Genie.png)

# Roadmap

- [ ] Implement the video-tokenizer. Use the MagViT-2 tokenizer as described in [Yu et al., (2023)](https://magvit.cs.cmu.edu/v2/).
- [ ] Implement the Latent Action Model, a Vector-Quantized ST-Transformer. Predict game-action from past video frames.
- [ ] Implement the Dynamics Model, which takes past frames and actions and produces the new video frame.
- [ ] Add functioning training script (Lightning).
- [ ] Show some results.

# Requirements

Code was tested with Python 3.11+ and requires `torch 2.0+` (because of use of fast flash-attention). To install the required dependencies simply run `pip install -r requirements.txt`

# Citations

This repo builds upon the beautiful MagViT implementation by [lucidrains](https://github.com/lucidrains/magvit2-pytorch/tree/main).

```bibtex
@article{bruce2024genie,
  title={Genie: Generative Interactive Environments},
  author={Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and Hughes, Edward and Lai, Matthew and Mavalankar, Aditi and Steigerwald, Richie and Apps, Chris and others},
  journal={arXiv preprint arXiv:2402.15391},
  year={2024}
}
```

```bibtex
@article{yu2023language,
  title={Language Model Beats Diffusion--Tokenizer is Key to Visual Generation},
  author={Yu, Lijun and Lezama, Jos{\'e} and Gundavarapu, Nitesh B and Versari, Luca and Sohn, Kihyuk and Minnen, David and Cheng, Yong and Gupta, Agrim and Gu, Xiuye and Hauptmann, Alexander G and others},
  journal={arXiv preprint arXiv:2310.05737},
  year={2023}
}
```
