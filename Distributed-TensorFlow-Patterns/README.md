# Advanced Distributed Deep Learning with TensorFlow 1.x 🚀

![TensorFlow](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview
This repository contains advanced implementations of distributed training patterns using **TensorFlow 1.x**. It demonstrates how to effectively utilize cluster specifications, parameter servers, and worker nodes to scale deep learning models. The experiments are conducted on the MNIST dataset and cover various distributed architectural choices.

## 📂 Project Architecture & Contents

The repository is divided into three main experimental phases:

### 1. Distributed Hyperparameter Grid Search (`01_Distributed_GridSearch/`)
Implements a Parameter Server (PS) architecture to perform a distributed grid search for hyperparameter tuning.
*   **Concepts:** `tf.train.ClusterSpec`, `tf.train.Server`, `tf.train.replica_device_setter`.
*   **Workflow:** Multiple worker nodes train independent DNN models with different hyperparameter sets (learning rate, batch size) in parallel, selecting the top 3 configurations based on validation accuracy.

### 2. Distributed Ensemble Learning (`02_Distributed_Ensemble/`)
Leverages Model Parallelism to create a robust ensemble model.
*   **Concepts:** Multi-device graph placement, `tf.device` context management, Ensemble averaging.
*   **Workflow:** The top 3 models from the previous phase are placed on separate worker devices within a single computational graph. Their outputs are averaged to produce a highly accurate, distributed ensemble prediction.

### 3. Sync vs. Async Updates & Model Parallelism (`03_Sync_vs_Async_Updates/`)
Explores the impact of gradient update strategies and vertical model slicing on training efficiency.
*   **Concepts:** `tf.train.SyncReplicasOptimizer`, `MonitoredTrainingSession`, Asynchronous gradients, Vertical Model Slicing.
*   **Workflow:** 
    *   Compares training time and accuracy between standard **Asynchronous** updates and **Synchronous** updates using `SyncReplicasOptimizer`.
    *   Implements **Vertical Model Parallelism** by splitting the DNN layers across multiple devices and evaluating the performance trade-offs.

---

## ⚙️ Installation & Requirements

To run these experiments, you need a Python environment with TensorFlow 1.x installed.
```bash
# Clone the repository
git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

# Install requirements
pip install -r requirements.txt
*(Note: TensorFlow 1.15.0 is highly recommended for these TF1 specific APIs).*

---

## 🚀 How to Run

Since these are distributed applications, you need to launch multiple terminal sessions to simulate or run on an actual cluster. 

**Example: Running a Parameter Server and Two Workers**

*Terminal 1 (Parameter Server):*
bash
python main.py --job_name=ps --task_index=0

*Terminal 2 (Worker 0):*
bash
python main.py --job_name=worker --task_index=0

*Terminal 3 (Worker 1):*
bash
python main.py --job_name=worker --task_index=1
*(Refer to the specific `README.md` inside each sub-folder for detailed running instructions for that experiment).*

---

## 📊 Key Findings & Results

*   **Grid Search:** Distributed training reduced hyperparameter search time by approximately `[X]%`.
*   **Ensemble:** The distributed ensemble achieved an accuracy of `[Y]%`, outperforming the best individual model by `[Z]%`.
*   **Sync vs. Async:** Asynchronous training was faster per step, but Synchronous training converged in fewer epochs. Vertical model parallelism showed a `[speed-up/slow-down]` due to communication overhead.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/[Your-Username]/[Your-Repo-Name]/issues).

## 📝 License
This project is [MIT](LICENSE) licensed.


***
