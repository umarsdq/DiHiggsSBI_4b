# Neural Simulation-based Inference for DiHiggs: Complete Workflow
This is the repository containing the code worked on during my UROP placement at Imperial College London in the CMS group. The majority of the work is adapted from the code used in paper "Constraining the Higgs Potential with Neural Simulation-based Inference for Di-Higgs Production" (https://arxiv.org/abs/2405.15847)

### General version notes
- Python 3.8 (PyTorch, NumPy etc.)
- MadMiner installation (custom [branch](https://github.com/umarsdq/madminer) which allows MadSpin cards)
- MadGraph [3.5.1](https://herwig.hepforge.org/downloads?f=mirror/MG5_aMC_v3.5.1.tar.gz) (Pythia8, Delphes 3.5.1, [SMEFT@NLO model](https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO), LHAPDF). 

Full instructions to install and configure the Madgraph 3.5.1 installation can be found [here](https://github.com/umarsdq/DiHiggsSBI_4b/blob/main/MG_install.md). For Imperial users, you may use the existing installation directory:

`/vols/cms/us322/mg5amcnlo`

## Setup working directory

For cluster users, it is recommended to first create a temporary directory for all pip cache 

```
export TMPDIR=/vols/cms/us322/tmp
```

To setup the workflow and create the conda environment,

```
conda create -n madminer_env python=3.8
conda activate madminer_env
git clone https://github.com/umarsdq/DiHiggsSBI_4b.git
cd DiHiggsSBI_4b
git clone https://github.com/umarsdq/madminer.git
cd madminer
pip install -e .
cd ..
pip install -r requirements.txt
python -m ipykernel install --user --name python38_madminer --display-name "Python3.8 (madminer)"
```

The directories must then be correctly configured. 

## Analysis flow

The file `workflow.yaml` is used for I/O management, so such folders don't have to be defined in every script. You must change the contents to your working directory. It is recommended to generate events within a temporary large disk space.

### Event generation
Most of the instructions below are provided by the [authors](https://github.com/rmastand/nsbi_for_dihiggs)

1. `01_setup_morphing_basis.ipynb`: specify the SMEFT operators (in the SMEFT@NLO basis) and define a set of "benchmark points". For every event generated in MadGraph, weights corresponding to each benchmark will be computed. 

2. `02_generate_events.py`: generate the events with MadGraph. MadGraph commands, MadSpin cards, and Pythia cards can be found in the `cards` folder. MadGraph run cards are in `cards/run_cards`.

   There are run cards for both the 14 TeV and 100 TeV collider setups. Signal runs cards have no cuts on the decay products, since MadGraph is only used for the $gg \rightarrow hh$ decays, and MadSpin in used for the higgs decays. Background run cards have stricter angular and mass window cuts corresponding to those specified in the main paper. 

   To generate signal events and background events, add the `-sm` and `-b` flags respectively. For BSM event generation, add the `-supp` flag followed by the BSM id (1-9). For example, to generate events at the non-SM benchmark 2, run `python 02_generate_events.py -supp -supp_id 2`. **You can specify the desired number of Madgraph runs in the `workflow.yaml`.**


4. `03a_read_delphes.py`: Run Delphes on the previously generated files and make selection cuts on the events. 

   This script assumes that you have a specific directory setup, namely that the outputs of step 2 are in `</path_from_workflow_yaml_delphes_input_dir_prefix/process_id/batch_<i>/`. `process_id` is an argument to the script (`signal_sm`, `signal_supp` for non-SM benchmarks, or `background_0`), and the batch is indexed by an integer. That directory can contain any number of Madgraph output directories `run_j`. 

   n.b. This directory setup must be manually created, but I have found that it works well when generating a large number of events. especially when events are generated in parallel on a cluster setup with separate scratch and long-term storage directories. 

   As an example, you could run Delphes and apply kinematic cuts on events from 20 MadGraph runs that have been generated at the non-SM benchmark 2 by running `python 03a_read_delphes.py -p signal_supp -supp_id 2 -b 0 -start 0 -stop 20`. 

   Finally, compile events over batches and all signal benchmarks with `python 03b_compile.py -p signal` and `python 03b_compile.py -p background`.

5. `04_make_samples.ipynb`: generate samples of signal events at arbitrary benchmark points, using MadMiner. These samples will be used for network training and testing. You can generate multiple datasets (identified by `parameter_code`) depending on which SMEFT Wilson coefficients you want to vary.


### Likelihood rato evaluation
5. `05_train_network.py`: Train the neural networks (classifiers). Specify the dataset that you want to run over be changing `sampling.output_dir` in `workflow.yaml` and by providing the correct `parameter_code` for the argument. Both simple dense nets and Bayesian nets are implemented. Network architecture and hyperparameters are hard-coded in the script, but they are all saved out into a config `yaml` with a particular run id (`rid`, specified in the arguments). 

   As described in the paper, there are 3 classifiers that need to be trained: (1) likelihood ratio of BSM signal to SM signal, (2) likelihood ratio of BSM signal to background, (3) likelihood ratio of background to SM signal. 

   As an example, you could train classifier 1 on a set of data that only varies over the first Wilson coefficient, using 5 kinematic features, with `python 05_train_network.py -p c0 -rid test_run -f 5 -c1`.

6. `06a_evaluate_test_statistic.ipynb` and `06b_evaluate_coverage.ipynb`: calculate likelihood ratios on previously generated test sets (or multiple likelihood ratios over different test set instatiations). It is possible to ensemble over several networks.

7. `07_nice_plots.ipynb`: Plot log-likelihood ratios and coverage tests for varying wilson coefficients.

Finally, `visualize_features.ipynb` is helpful to quickly visualize how kinematic features change as a function of Wilson coefficients.