# RD-Suite: A Benchmark for Ranking Distillation

RD-Suite provides a dataset for ranking distillation research.

## Evaluation Script
RD_Suite_Eval_Script.ipynb uses open-sourced operations to reproduce teacher
ranker metrics, using the TREC data formats provided below.

## Data
We provide data in TREC format. The evaluation script automatically downloads
rd_suite_test.zip, which contains dev/test data for all tasks, and can be used
to run the evaluation script end-to-end. The other 4 datasets are self-contained
as listed below. For NQ, we also provide teacher scores from a ranker trained on
 MSMARCO for the distillation transfer task.

| Dataset |   Size  | Link  |
|:---------------:|:------:|:-----------:|
| MSMARCO | 513M | [msmarco.zip](https://storage.googleapis.com/gresearch/rd-suite/msmarco.zip)  |
| NQ | 125M | [nq.zip](https://storage.googleapis.com/gresearch/rd-suite/nq.zip)  |
| Web30K | 65M | [web30k.zip](https://storage.googleapis.com/gresearch/rd-suite/web30k.zip)  |
| Istella | 173M | [istella.zip](https://storage.googleapis.com/gresearch/rd-suite/istella.zip)  |


## Citation
You can cite our work by copying the following bibtex:

```
@article{qin2023rd,
  title={RD-Suite: A Benchmark for Ranking Distillation},
  author={Qin, Zhen and Jagerman, Rolf and Pasumarthi, Rama and Zhuang, Honglei and Zhang, He and Bai, Aijun and Hui, Kai and Yan, Le and Wang, Xuanhui},
  journal={arXiv preprint arXiv:2306.04455},
  year={2023}
}
```

