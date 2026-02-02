# Machine Learning Project – Q-Learning on Taxi-v3

This project implements a **tabular Q-learning Reinforcement Learning agent** applied to the **Taxi-v3** environment from *Gymnasium*. The goal is to learn an **optimal policy** that allows the taxi to correctly pick up and drop off a passenger while minimizing the number of steps and maximizing the total reward.

---

## 📌 Problem Description

The **Taxi-v3** environment is a classic *Markov Decision Process (MDP)* with:

* **500 discrete states** (taxi position, passenger position, destination)
* **6 possible actions**:

  * South
  * North
  * West
  * East
  * Pickup
  * Dropoff

### Reward Function

* **+20**: successful passenger drop-off
* **-1**: for each step taken
* **-10**: illegal pickup or drop-off

The agent must therefore learn to complete the task in the **minimum number of steps**.

---

## 🧠 Approach

The project uses **tabular Q-learning**, a *model-free* Reinforcement Learning algorithm.

The Q-table update rule is:

[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\right]
]

Where:

* `α` is the *learning rate*
* `γ` is the *discount factor*
* `r` is the immediate reward
* `s'` is the next state

### Exploration Strategy

An **epsilon-greedy policy** is used:

* with probability `ε`, the agent explores
* with probability `1 − ε`, the agent exploits current knowledge

The value of `ε` decreases over time through *epsilon decay*.

---

## ⚙️ Project Structure

```
MLProjectMario.py
```

### Main Components

* **`Agent` class**

  * Manages the Q-table
  * Action selection (epsilon-greedy)
  * Q-learning update rule

* **`train_agent`**

  * Trains the agent in the environment
  * Progressively fills the Q-table

* **`evaluate_agent`**

  * Evaluates the learned policy without exploration

* **`plot_training_results`**

  * Displays:

    * Reward per episode
    * Moving average of rewards
    * Epsilon decay

* **`run_demo`**

  * Runs a rendered episode using the greedy policy

---

## 📈 Results

During training, the following behavior is observed:

* **First ~1000 episodes**: average reward between `-160` and `-200`

  * expected behavior due to high exploration

* **Around 2000 episodes**:

  * the agent learns the structure of the problem

* **Final phase**:

  * average reward ≈ **7.3 – 7.5**
  * corresponding to the **optimal policy**

This value is consistent with the reward function:

```
20 - ~12/13 steps ≈ 7.3
```

---

## ▶️ How to Run

### Requirements

* Python ≥ 3.9
* gymnasium
* numpy
* matplotlib

Install dependencies:

```bash
pip install gymnasium numpy matplotlib
```

### Execution

```bash
python MLProjectMario.py
```

The script will:

1. Train the agent
2. Evaluate its performance
3. Display training plots
4. Run a rendered demo episode

---

## 🎯 Learning Objectives

* Understand how **tabular Q-learning** works
* Analyze the effect of `epsilon`, `alpha`, and `gamma`
* Observe the transition from exploration to exploitation
* Visualize convergence toward an optimal policy

---

## 📚 Technologies Used

* Python
* Gymnasium
* NumPy
* Matplotlib

---



