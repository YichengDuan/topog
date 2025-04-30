# A Topological Graph Framework for Enhanced Scene Detection and Understanding in Robotic Navigation


---

## 📖 Table of Contents

- [⚙️ Requirements](#️-requirements)  
- [📥 Installation](#-installation)  
- [🚦 Quick Start](#-quick-start)  
- [🛠 Configuration](#-configuration)  
- [📁 Repository Structure](#-repository-structure)  
- [🔍 Example Usage](#-example-usage)  
- [📈 Evaluation](#-evaluation)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  
- [📚 Citation](#-citation)


---

## ⚙️ Requirements

This project depends on [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and the Matterport3D (MP3D) dataset. Please follow the official installation instructions in their respective repositories to set them up properly.

Additional Python dependencies include:

- `matplotlib`
- `numpy`
- `pandas`
- `transformers`
- `torch`
- `torch_geometric`

Make sure to install these packages via `pip` or `conda` before running the project.

---

## 🔍 Example Usage


```bash
# Build the topological graph
python ./build_raw_graph_data.py

# Plot topological graph statistics
python ./graph_stats.py

# Train GCN models
python ./leiden_GCN_train.py
python ./leiden_GINE_train.py
python ./leiden_SAGE_train.py

# Run ablation study and generate plots
python ./ablation_study.py
python ./ablation_study_plot.py
```
---

## 📈 Evaluation

- Topological graph visualizations can be found in the `./data/` directory.
- All experimental results are saved in the `./results/` directory.

## 🤝 Contributing

1. Fork → Clone → Create feature branch  
2. Add tests for new modules  
3. Submit PR → Review → Merge  

---

## 📄 License

This project is MIT Licensed. See `LICENSE` for details.

---

## 📚 Citation


If you find our work helpful, feel free to give us a cite:
```
@misc{topoLGCN,
    title = {A Topological Graph Framework for Enhanced Scene Detection and Understanding in Robotic Navigation},
    url = {https://github.com/YichengDuan/topog},
    author = {Yicheng Duan, Duo Chen},
    month = {April},
    year = {2025}
}
```

