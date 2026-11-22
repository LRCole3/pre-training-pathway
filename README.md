# Pre-Training, From First Principles: n-grams → Transformers → Diffusion → Diffusion Transformers

A progressive, hands-on series on **pre-training**: how we got from counting word pairs to the base models that
today's systems start from. It is the prequel to the
[post-training pathway](https://github.com/LRCole3/post-training-pathway), which begins where this one ends.

Three rules shape every notebook:

* **Each method appears because the previous one fails at something measurable.** A shared scorecard
  (`results/scorecard.csv`) makes the failures visible.
* **Design choices are explained, not just used.** Every notebook has a *Design choices: what changed since* section:
  the original choice, what the field moved to (rotary position embeddings, mixture of experts, flow matching, ...),
  why, and an ablation where a laptop can show the effect. Where it cannot, the notebook says so.
* **Visual first, interactive where possible.** Diagrams, heat maps and animations for each concept, plus
  slider-driven explorers.

## Why toy-sized models?

Real pre-training needs data centres. This series instead builds two small worlds with a **ground truth**:

* **Language (01–05, 10):** a synthetic language with a known generator (`prelib/world.py`). Words have classes,
  documents have a hidden topic and a hidden style, and a document can bind a key to a value and ask for it later.
  Because we own the generator, the **best achievable loss is computed exactly**, long-range recall can be tested at
  any distance, and the hidden style gives a downstream task for measuring what pre-training learned.
* **Images (06–10):** scikit-learn's 8×8 handwritten digits (no download), scored by a frozen "judge" classifier
  that plays the role of FID's Inception network.

The trade-off: these notebooks show **mechanisms and failure modes**, not the papers' headline numbers. Several
modern design choices pay off only at scales a laptop cannot reach. Results are reported as measured.

## Learning path

| # | Notebook | Methods | Design choices covered | Limitation that motivates the next step | Status |
|---|---|---|---|---|---|
| 01 | [Counting and embeddings](01_ngrams_and_embeddings.ipynb) | n-grams, smoothing, window MLP, skip-gram | Kneser-Ney vs add-k; negative sampling; subword vectors | Fixed window; one vector per word | available |
| 02 | [Recurrent models, attention, contextual pretraining](02_rnns_attention_elmo.ipynb) | RNN, LSTM, GRU, attention, ELMo-style transfer | Gating; gradient clipping; additive vs dot-product attention | Memory fades with distance; no parallel training | available |
| 03 | [The transformer, built from scratch](03_transformer_from_scratch.ipynb) | Self-attention, the 2017 block, then today's block | Pre-LN and RMSNorm; RoPE and ALiBi; SwiGLU; grouped-query attention; mixture of experts | Needs an objective and a transfer recipe | available |
| 04 | [Pretraining objectives and transfer](04_objectives_and_transfer.ipynb) | Causal, masked, span corruption; probing and fine-tuning | Masking rate; dynamic and span masking; encoder vs decoder | No objective wins everything | available |
| 05 | [Making pretraining scale](05_scaling_pretraining.ipynb) | Tokenisers, optimiser recipe, scaling laws, in-context recall | BPE and vocabulary size; AdamW; warmup and cosine decay; weight tying | Discrete tokens only | coming next |
| 06 | [Generative pretraining for images before diffusion](06_vae_gan_autoregressive.ipynb) | Autoencoder, VAE, GAN, WGAN-GP, pixel autoregression | KL weight and annealing; GAN losses and penalties | Blur, mode collapse, slow sampling | coming next |
| 07 | [The diffusion training design space](07_diffusion_design_space.ipynb) | DDPM and its training choices | Schedules; prediction targets; loss weighting; preconditioning; EMA | Pixel space, MLP backbone, no text | coming next |
| 08 | [Pretrained parts: encoders, CLIP, latent diffusion](08_representations_and_latent_diffusion.ipynb) | MAE, contrastive learning, CLIP-style model, latent diffusion | Mask ratio; softmax vs sigmoid contrastive loss; KL vs VQ latents | The backbone still does not scale | coming next |
| 09 | [Pretraining diffusion transformers](09_pretraining_diffusion_transformers.ipynb) | DiT, flow matching, scaling, training accelerators | Conditioning route; position scheme; timestep sampling; masking; REPA | Base models still need post-training | coming next |
| 10 | [Capstone: one architecture, three objectives](10_capstone_unified_pretraining.ipynb) | Autoregressive vs masked vs diffusion, for text and images | Generation order; masking schedule; discrete vs continuous tokens | The post-training series starts here | coming next |

## Getting started

```bash
git clone https://github.com/LRCole3/pre-training-pathway.git
cd pre-training-pathway
pip install -r requirements.txt
jupyter lab
```

Run the notebooks **in order, from the repository root**. No GPU is needed.

**Interactive parts.** The sliders need a running notebook (JupyterLab, VS Code or Colab). GitHub's viewer shows a
static preview of each explorer at its default settings, and the animations (GIFs in `figures/`) play everywhere.
Each notebook has an "Open in Colab" badge; that path has not been tested by the author.

## Repository layout

```
├── 01_ … 10_*.ipynb
├── prelib/
│   ├── world.py      # the synthetic language, its oracle loss floor, recall tests and the probe task
│   ├── viz.py        # figure style, diagrams, heat maps, token strips, GIF animations, explore() sliders
│   ├── images.py     # 8x8 digits, judge classifier, sample grids
│   └── metrics.py    # shared scorecard (language and image metrics)
├── figures/          # animations produced by the notebooks
├── research_papers/  # every cited paper, by notebook, with verified arXiv / DOI links
├── results/scorecard.csv
└── requirements.txt
```

Each notebook follows the same format: an overview that places it in the series, learning objectives, worked
sections with figures, a design-choices section, **Challenge Problems** with a solution cell immediately below,
a final summary and references with links.

## Credit

The teaching format follows the
[Protein_Design `ppi_deep_learning` series](https://github.com/LRCole3/Protein_Design/tree/main/ppi_deep_learning).
`prelib/images.py` and `prelib/metrics.py` are reused from the post-training pathway. The methods belong to the
authors cited in each notebook; see [`research_papers/README.md`](research_papers/README.md).
