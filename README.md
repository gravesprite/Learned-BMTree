

# Learned-BMTree

This is the open codebase for the project: Learned BMTree

### Requirement

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



