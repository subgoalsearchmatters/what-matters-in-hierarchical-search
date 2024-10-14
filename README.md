# What Matters in Hierarchical Search for Combinatorial Reasoning Problems?

Combinatorial reasoning problems, particularly the notorious NP-hard tasks, remain a significant challenge for AI research. A common approach to addressing them combines search with learned heuristics. Recent methods in this domain utilize hierarchical planning, executing strategies based on subgoals. Our goal is to advance research in this area and establish a solid conceptual and empirical foundation. 
Specifically, we identify the following key obstacles, whose presence favors the choice of hierarchical search methods: **hard-to-learn value functions, complex action spaces, presence of dead ends in the environment** or **training data collected from diverse sources**. Through in-depth empirical analysis, we establish that hierarchical search methods consistently outperform standard search methods across these dimensions, and we formulate insights for future research. On the practical side, we also propose a consistent evaluation guidelines to enable meaningful comparisons between methods and reassess the state-of-the-art algorithms.

Our main contributions are the following:
- We present a comprehensive empirical analysis comparing the performance of hierarchical search methods against low-level search methods across diverse problem settings.
- We identify problem characteristics that influence performance, providing insights into when hierarchical methods should be favored over low-level methods.
- We propose a standardized evaluation guidelines that facilitate meaningful and consistent comparisons across different types of search methods
This repository contains code essential for reproducing results presented in our paper. 


## Datasets
For fair and transparent evaluation, we share following datasets on which we have conducted our experiments:
- Sokoban datasets: https://drive.google.com/drive/folders/1tnW9thS5CKdxB6ZA0RumbUjb8eYEtbie?usp=sharing
- INT datasets: https://drive.google.com/drive/folders/1uJMMZeEUOn4kYAettY0EkZWoN8cjcPNA?usp=sharing
- Rubik's Cube datasets: https://drive.google.com/drive/folders/1adgNwQsEK3fnBa9lpvmx74W5jmOiB18Y?usp=sharing
- NPuzzle datasets: https://drive.google.com/drive/folders/1WL4F35XvayTtDhGBkM3Yc2AR56xxn5cS?usp=sharing

## Running code

To run the code locally, simply run the command below

`python3 -m carl.run --config_file config_file_name.yaml`

## Solving problem instances

Below there is an example of MCTS config on sokoban environment:

```yaml
params:
  env:
    _target_: carl.environment.sokoban.env.SokobanEnv
    tokenizer:
      _target_: carl.environment.sokoban.tokenizer.SokobanTokenizer
      cut_distance: 150
      type_of_value_training: "regression"
      size_of_board:
        - 12
        - 12
    num_boxes: 4

algorithm:
  _target_: carl.solve_instances.solve_instances.SolveInstances

  # solver class
  solver_class:
    _target_: carl.solver.mcts.MCTS
    number_of_steps: 200
    number_of_simulations: 5
    discount: 1
    sampling_temperature: 1
    env: ${params.env}

    value_function:
      _target_: carl.inference_components.mcts_value_wrapper.MCTSValueWrapper
      value:
        _target_: carl.inference_components.value.TransformerValue
        value_network:
          _target_: transformers.BertForSequenceClassification.from_pretrained
          _partial_: true
        path_to_value_network_weights: "./validation/sokoban/components/full_data/value/checkpoint-1343100"
        type_of_evaluation: "regression"
        noise_variance: 0
        env: ${params.env}

    policy_state_builder:
      _target_: carl.mcts_states_builder.mcts_sokoban_states_builder.StateBuilderSokoban
      policy:
        _target_: carl.inference_components.policy.TransformerPolicy
        policy_network:
          _target_: transformers.BertForSequenceClassification.from_pretrained
          _partial_: true
        path_to_policy_weights: "./validation/sokoban/components/full_data/policy/checkpoint-94820"
        env: ${params.env}
        n_actions: 2
      env: ${params.env}

  data_loader_class:
    _target_: carl.environment.instance_generator.BasicInstanceGenerator
    generator:
      _target_: carl.environment.instance_generator.GeneralIterableDataLoader
      path_to_folder_with_data: "./validation/sokoban/progress/deep_mind_12_12_4"
    batch_size: 64

  # result logger
  result_logger:
    _target_: carl.solve_instances.result_loggers.MCTSResultLogger
    custom_logger:
      _target_: carl.custom_logger.loggers.NeptuneCaRLLogger
      name: "mcts_sokoban_big_budget"
      description: "MCTS for sokoban_big_budget"
      project: ""
      tags: "mcts_sokoban_big_budget"
      log_parameters: False
    budget_logs: [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000, 2000, 2500, 3000, 4000, 5000, 10000,  50000, 100000, 500000, 1000000, 5000000]
    problem_to_solve: 100
  problem_to_solve: 100

  n_parallel_workers: 1

carl_workers:
  solve:
    algorithm._target_: carl.solve_instances.solve_instances.SolveInstances

carl_grid:
  - algorithm.problem_to_solve: [100]
    algorithm.solver_class.number_of_steps: [50, 100]
    algorithm.solver_class.number_of_simulations: [100, 500, 1000]
    algorithm.solver_class.discount: [0.99]
    algorithm.solver_class.sampling_temperature: [0, 0.5, 1]
    algorithm.solver_class.value_function.value.noise_variance: [0]
```

In order to run following experiments, please use following command:

```bash
python3 -m carl.run --config-dir experiments --config-name sokoban_ood_10b_mcts_solve hydra/launcher=basic --multirun
```

## Training components

To train components use following configs (example for sokoban environment):
- value function `sokoban_train_value.yaml`
- subgoal generator `sokoban_train_generator.yaml`
- conditional low-level policy `sokoban_train_cllp.yaml`
- behavioral cloning policy `sokoban_train_policy.yaml`

To train components use following command:

```bash
python3 -m carl.run --config-dir components --config-name sokoban_train_value hydra/launcher=basic --multirun
```