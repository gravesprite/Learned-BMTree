

# Learned-BMTree

This is the open codebase for the project: Learned BMTree

## Requirement

PostgreSQL: https://www.postgresql.org/download/

Python Environment: 

```shell
conda env create -f environment.yml
```



## Training and PGTest

Training with the uniform dataset and skew query workload:

```shell
# Training
nohup python exp_opt_fast.py --data uniform_1000000 --query skew_1000_dim2 &
```

Testing BMTree under PostgreSQL:

```shell
# PostgreSQL Test
python pg_test.py --pg_test_method bmtree  --data uniform_1000000 --query skew_2000_dim2 --bmtree mcts_bmtree_uni_skew_1000 --db_password ''
```

## Citation

Kindly cite our paper if you find it helpful:
```bibtex
@inproceedings{bmtree2023,
  title={Towards Designing and Learning Piecewise Space-Filling Curves},
  author={Li, Jiangneng and Wang, Zheng and Cong, Gao and Long, Cheng and Kiah, Han Mao and Cui, Bin},
  journal={Proceedings of the VLDB Endowment},
  volume={16},
  number={9},
  year={2023},
  publisher={VLDB Endowment}
}
```

