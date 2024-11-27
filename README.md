<h2 align="center"> <a href="">Identity-Preserving Text-to-Video Generation by Frequency Decomposition</a></h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>


<h5 align="center">


[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space)
[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/BestWishYsh/)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.17440-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17440) 
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://pku-yuangroup.github.io/ConsisID/) 
[![Dataset](https://img.shields.io/badge/Dataset-previewData-green)](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/ConsisID/blob/main/LICENSE) 

</h5>

<div align="center">
This repository is the official implementation of ConsisID, a tuning-free DiT-based controllable IPT2V model to keep human-identity consistent in the generated video. The approach draws inspiration from previous studies on frequency analysis of vision/diffusion transformers.
</div>






<br>

<details open><summary>💡 We also have other video generation project that may interest you ✨. </summary><p>
<!--  may -->



> [**Open-Sora-Plan**](https://github.com/PKU-YuanGroup/Open-Sora-Plan) <br>
> PKU-Yuan Lab and Tuzhan AI etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)  <br>
>
> [**MagicTime**](https://arxiv.org/abs/2404.05014) <br>
> Shenghai Yuan, Jinfa Huang and Yujun Shi etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/MagicTime)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/MagicTime.svg?style=social)](https://github.com/PKU-YuanGroup/MagicTime) <br>
>
> [**ChronoMagic-Bench**](https://arxiv.org/abs/2406.18522) <br>
> Shenghai Yuan, Jinfa Huang and Yongqi Xu etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/ChronoMagic-Bench/)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/ChronoMagic-Bench.svg?style=social)](https://github.com/PKU-YuanGroup/ChronoMagic-Bench/) <br>
> </p></details>


## 📣 News

* ⏳⏳⏳ Release the full codes & datasets &  weights.
* ⏳⏳⏳ Integrate into Diffusers, ComfyUI, Replicate and Jupter-notebook.
* `[2024.11.27]`  🔥 We release the arXiv paper for ConsisID, and you can click [here](https://arxiv.org/abs/2411.17440) to see more details.
* `[2024.11.22]`  🔥 **All codes & datasets** are coming soon! Stay tuned 👀!

## 😍 Gallery

Identity-Preserving Text-to-Video Generation.

[![Demo Video of ConsisID](https://github.com/user-attachments/assets/634248f6-1b54-4963-88d6-34fa7263750b)](https://www.youtube.com/watch?v=PhlgC-bI5SQ)
or you can click <a href="https://github.com/SHYuanBest/shyuanbest_media/raw/refs/heads/main/ConsisID/showcase_videos.mp4">here</a> to watch the video.

## 🤗 Demo

### Gradio Web UI

Highly recommend trying out our web demo by the following command, which incorporates all features currently supported by ConsisID. We also provide [online demo](https://huggingface.co/spaces/BestWishYsh/ConsisID-preview-Space) in Hugging Face Spaces.

```bash
python app.py
```

### CLI Inference

```bash
python infer.py --model_path BestWishYsh/ConsisID-preview
```

warning: It is worth noting that even if we use the same seed and prompt but we change a machine, the results will be different.

## ⚙️ Requirements and Installation

We recommend the requirements as follows.

### Environment

```bash
git clone --depth=1 https://github.com/PKU-YuanGroup/ConsisID.git
cd ConsisID
conda create -n consisid python=3.11.0
conda activate consisid
pip install -r requirements.txt
```

### Download ConsisID

```bash
# model hub: https://huggingface.co/BestWishYsh/ConsisID-preview
# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type model \
BestWishYsh/ConsisID-preview \
--local-dir BestWishYsh/ConsisID-preview
```

## 🗝️ Training

#### Data preprocessing

```
coming soon!
```

#### Video DiT training

Setting hyperparameters

```
coming soon!
```

Then, we run scripts/train.sh.

```bash
# For single rank
bash train_single_rank.sh
# For multi rank
bash train_multi_rank.sh
```

## 👍 Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [CogVideoX](https://github.com/THUDM/CogVideo), [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun).


## 🔒 License

* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/ConsisID/blob/main/LICENSE) file.
* The CogVideoX-5B model (Transformers module) is released under the [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
* The service is a research preview. Please contact us if you find any potential violations. (shyuan-cs@hotmail.com)

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@misc{ConsisID,
      title={Identity-Preserving Text-to-Video Generation by Frequency Decomposition}, 
      author={Shenghai Yuan and Jinfa Huang and Xianyi He and Yunyuan Ge and Yujun Shi and Liuhan Chen and Jiebo Luo and Li Yuan},
      year={2024},
      eprint={2411.17440},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17440}, 
}
```

## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/ConsisID/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/ConsisID&anon=true" />

</a>

