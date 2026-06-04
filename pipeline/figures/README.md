# 论文图表 — 脚本与输出对照

## 主文图

| Figure | 脚本 | 输出 | 内容 |
|--------|------|------|------|
| Fig.1 | (手绘) | — | 模型架构 |
| Fig.2 | fig2_benchmark.py | Fig2_benchmark/Figure2.png | 主实验 benchmark |
| Fig.3 | fig3_biological.py | Fig3_biological/Figure3.png | UMAP+recall+marker |
| Fig.4 | (inline script) | Fig4_enrichment/Figure4a_GO.png, Figure4b_KEGG.png | GO/KEGG 富集 |
| Fig.5 | fig5_ablation.py | Fig5_ablation/Figure5.png | 消融实验 |
| Fig.6 | fig6_sensitivity.py | Fig6_sensitivity/Figure6.png | 参数敏感性 |
| Fig.7 | fig7_robustness.py | Fig7_robustness/Figure7.png | 下采样鲁棒性 |
| Fig.8 | fig8_scalability.py | Fig8_scalability/Figure8.png | 可扩展性 |

## 补充材料

| Figure | 脚本 | 输出 |
|--------|------|------|
| Fig.S1 | figS1_similarity.py | FigS1_similarity/FigS1.png |

## 附加素材(marker gene)

| 文件 | 内容 |
|------|------|
| Fig7_marker/Figure7_matrixplot.png | Marker gene matrix plot |
| Fig7_marker/Figure7_dotplot.png | Marker gene dot plot |

## 一键重新生成
```bash
cd /home/liyang/BioJiaheWang/scAGCR
python pipeline/figures/fig2_benchmark.py
python pipeline/figures/fig3_biological.py
python pipeline/figures/fig5_ablation.py
python pipeline/figures/fig6_sensitivity.py
python pipeline/figures/fig7_robustness.py
python pipeline/figures/fig8_scalability.py
python pipeline/figures/figS1_similarity.py
```
