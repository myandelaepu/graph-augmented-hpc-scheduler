# Graph-Augmented Deep Reinforcement Learning for Energy-Efficient HPC Job Scheduling
A comprehensive framework that combines Graph Attention Networks (GAT) with Deep Reinforcement Learning (DRL) to optimize energy consumption and performance in High-Performance Computing (HPC) environments. This implementation provides a novel approach to job scheduling that considers complex job dependencies, resource relationships, and energy efficiency across multiple supercomputing platforms.
# Overview
This repository implements a scheduling framework that leverages graph neural networks to model inter-job dependencies and resource relationships in HPC systems. The approach achieves significant energy reductions while maintaining high system throughput and resource utilization across diverse computational workloads.
# Key Features
1.	Graph Attention Networks: Captures complex job dependencies and resource relationships through multi-head attention mechanisms
2.	Multi-Objective Optimization: Balances energy efficiency with system performance metrics including throughput and utilization
3.	Real-World Validation: Tested on actual workload traces from Polaris, Mira, and Cooley supercomputing systems
4.	Scalable Architecture: Demonstrated linear complexity scaling up to 49,152 nodes
5.	Comprehensive Energy Modeling: Physics-based power consumption models covering compute, memory, network, and cooling components
# Architecture
Core Components
Temporal Graph Attention Module
1.	Multi-head attention mechanism for modeling job dependencies
2.	Temporal embedding layers for time-series pattern recognition
3.	Dynamic graph construction based on project relationships and resource conflicts
Actor-Critic Network Architecture
1.	Policy network for scheduling decision generation
2.	Value network for state evaluation and reward estimation
3.	Entropy regularization for exploration-exploitation balance
# Energy-Aware Environment
1.	Physics-based power consumption modeling
2.	Real-time resource utilization tracking
3.	Multi-dimensional reward function incorporating energy and performance metrics
Scheduling Policy Options
1.	FIFO (First-In-First-Out)
2.	SJF (Shortest Job First)
3.	Priority-based scheduling with capability awareness
4.	Adaptive load-balancing strategies
# Performance Results
The framework demonstrates consistent improvements in Energy Efficiency, Throughput, and Resource Utilization across multiple HPC systems (Polaris System, Mira System,	Cooley System).
Installation and Dependencies
System Requirements
1.	Python 3.8 or higher
2.	CUDA-compatible GPU (optional but recommended for training)
3.	Minimum 8GB RAM (16GB recommended for large datasets)
Required Packages
# Core deep learning framework
pip install torch torchvision torchaudio

# Graph neural network components
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

# Data processing and analysis
pip install numpy pandas matplotlib seaborn scikit-learn

# Reinforcement learning environment
pip install gymnasium stable-baselines3

# Optimization and hyperparameter tuning
pip install wandb optuna ray[tune] bayesian-optimization

# Visualization and network analysis
pip install networkx plotly kaleido
Usage
Basic Implementation
from GAT_DRL_Scheduler import *

# Initialize system configuration
config = SystemConfig()

# Load workload datasets
workloads = load_workload_data()

# Train the GAT-DRL scheduler
results = train_gat_drl_scheduler(workloads, config, num_episodes=200)

# Evaluate against baseline schedulers
evaluation_results = evaluate_all_schedulers(workloads, config)
Custom Configuration
# Define custom system parameters
config = SystemConfig(
    polaris_nodes=560,
    polaris_cores_per_node=64,
    polaris_memory_per_node=512,
    polaris_gpu_per_node=4,
    polaris_pue=1.1
)

# Initialize environment with specific workload
env = HPCSchedulingEnvironment(workload_df, "polaris", config)

# Create and train scheduler
scheduler = GATDRLScheduler(input_dim=6, hidden_dim=128, lr=0.0003)
Dataset Information
Supported Workload Traces
1.	ANL Polaris: 560-node GPU cluster workload data (2024)
2.	ANL Mira: 49,152-node Blue Gene/Q system traces (2019)
3.	ANL Cooley: 126-node analysis cluster workload (2019)
Data Preprocessing
The framework includes comprehensive data preprocessing capabilities:
1.	Automatic handling of missing values and data type conversion
2.	Resource constraint validation and normalization
3.	Temporal feature extraction and job dependency analysis

Cooling and Infrastructure
1. Data center Power Usage Effectiveness (PUE) modeling
2.	Temperature-dependent cooling requirements
3.	Facility overhead calculations
Graph Construction Algorithm
Jobs are represented as nodes with comprehensive feature vectors:
1.	Resource Requirements: Nodes, cores, memory, storage
2.	Runtime Characteristics: Expected duration, historical patterns
3.	Project Affiliations: User groups and allocation categories
4.	Priority Levels: Queue priorities and capability requirements
Edges represent various dependency types:
1.	Project Relationships: Jobs from same research groups
2. 	Resource Conflicts: Competing resource requirements
3.	Temporal Correlations: Time-based scheduling dependencies
Training Process
1.	State Representation: Dynamic graph construction from current job queue and system state
2.	Action Selection: Multi-head attention over job dependencies with policy sampling
3.	Environment Interaction: Job scheduling decisions and resource allocation
4.	Policy Updates: Actor-critic learning with entropy regularization and gradient clipping
Code Organization
src/
├── models/
│   ├── temporal_graph_attention.py  
│   ├── actor_critic.py              
│   └── energy_model.py              
├── environment/
│   ├── hpc_scheduling_env.py      
│   └── workload_processing.py      
├── schedulers/
│   ├── gat_drl_scheduler.py        
│   └── baseline_schedulers.py
├── evaluation/
│   ├── performance_metrics.py      
│   └── statistical_analysis.py     
└── utils/
    ├── system_config.py          
    └── data_generation.py          



