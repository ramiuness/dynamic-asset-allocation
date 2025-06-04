+# 📊 Dynamic Portfolio Allocation under Uncertainty

This project implements and compares dynamic portfolio optimization methods for allocating wealth between risky and a risk-free asset over multiple periods. 
It explores both analytical and approximate solutions under different investor utility functions, including log and CRRA utilities.

---

## 🔍 Project Overview

- **Objective**: Determine optimal portfolio weights over a finite investment horizon in the presence of uncertainty.
- **Frameworks Used**:
  - Monte Carlo simulation
  - Convex optimization via CVXPY
  - Approximate Dynamic Programming (ADP)
- **Key Features**:
  - Exact dynamic programming with log utility and multi-asset convex optimization 
  - Approximate dynamic programming with a single risky asset, CRRA utility, and cubic interpolation
  - Wealth grid construction and policy extraction
---

## 📌 Modeling Assumptions

- The market is assumed to be **frictionless**, with no transaction costs, slippage, or liquidity constraints.
- Risk-free and risky assets are perfectly divisible.

---

## 📁 Contents

- `dynamicPortfolioAllocation.py`: Core module with reusable functions for:
  - Return simulation
  - Optimal portfolio computation
  - Value function approximation
- `dynamic_portfolio_allocation.ipynb`: Jupyter notebook demonstrating:
  - Examples of portfolio allocation under various scenarios
  - Visualizations of value functions and optimal policies
  - Comparison between exact and approximate solutions

---

## 🛠 Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![CVXPY](https://img.shields.io/badge/CVXPY-34495E?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)

---

## 📈 Key Results

- Demonstrates how optimal asset allocations change over time and across wealth levels.
- Highlights the performance of approximate dynamic programming versus exact methods.
- Offers a blueprint for extending the ADP to multi-asset, multi-period portfolio problems.

---

## 📚 Citation

If you use this code in your work, please cite this repository:

https://github.com/ramiuness/dynamic-portfolio-allocation

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

