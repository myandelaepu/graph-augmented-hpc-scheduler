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

# Installation and Dependencies

## System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for training)
- Minimum 8GB RAM (16GB recommended for large datasets)

## Installation
pip install -r requirements.txt

## Quick Start
The framework provides GAT-DRL schedulers for HPC job scheduling with energy optimization. Initialize the system with your workload data and begin training:

from GAT_DRL_Scheduler import SystemConfig, HPCSchedulingEnvironment

# Configure your HPC system parameters
config = SystemConfig()
env = HPCSchedulingEnvironment(workload_data, system_type="polaris", config=config)

See the `examples/` directory for complete implementation details and usage patterns.
