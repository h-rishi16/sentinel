# How Sentinel Works: A Plain-English Guide

If you are a recruiter, a hiring manager, or a developer exploring this repository, this document explains exactly **what** this platform does, **how** the math works, and **why** it was built this way—without assuming you have a Ph.D. in Quantitative Finance.

---

## 1. The Big Picture

At its core, **Sentinel** is a digital "Chief Risk Officer" for a hypothetical bank or hedge fund. 

Financial institutions have massive amounts of data flowing in every second: people swiping credit cards, loans being issued, and stock markets fluctuating. Sentinel’s job is to catch bad actors (Fraud), predict who won't pay us back (Credit Risk), and calculate how much money the bank might lose if the stock market crashes tomorrow (Market Risk).

Instead of having a separate app for each of these tasks, Sentinel unifies them into a single, high-speed dashboard.

### The Technology Under the Hood
* **The Backend (Python + FastAPI):** The brain. It crunches the numbers, runs the machine learning models, and talks to the database.
* **The Database (SQLite):** The memory. It stores thousands of synthetic borrowers, loans, and credit card swipes.
* **The Frontend (HTMX + Tailwind CSS):** The face. Instead of using a complex Javascript framework like React, Sentinel uses HTMX to magically swap parts of the HTML page in real-time straight from the Python server. This makes the app incredibly fast and lightweight.

---

## 2. Where Does the Data Come From? (The Data Forge)

Machine Learning models are useless without data. Because actual bank data is highly illegal to share, Sentinel includes a **Data Forge** (`scripts/seed_db.py`). 

When you click "Generate Data" in the UI, Sentinel acts like a simulation video game. It creates thousands of fake people, assigns them jobs, gives them credit scores, issues them loans, and simulates them swiping their credit cards over time. It even specifically programs some of these fake people to act like fraudsters so the AI has bad guys to learn from.

---

## 3. The 5 Risk Engines Explained

Once the data is generated, Sentinel uses 5 distinct mathematical engines to analyze it.

### Engine 1: Transaction Surveillance (Fraud Detection)
* **The Goal:** Stop a fraudulent credit card swipe the moment it happens.
* **How it works:** We use an AI algorithm called **XGBoost**. Rather than just looking at the amount of money spent, the algorithm looks at *behavior*. 
* **The Math:** Sentinel creates "Temporal Features" (e.g., did this person just swipe their card 5 times in the last hour?) and "Z-Scores" (e.g., is a $500 purchase statistically normal for this specific user, or wildly out of character?). The AI learns these patterns and assigns a fraud probability score to every swipe.

### Engine 2: Loan Default Engine (Credit Risk)
* **The Goal:** Decide if a borrower is too risky to give a loan to.
* **How it works:** This engine calculates the **Expected Loss (EL)** of a portfolio.
* **The Math:** It uses the classic banking formula: `Expected Loss = PD × LGD × EAD`.
  * **PD (Probability of Default):** The AI looks at their FICO score and Debt-to-Income (DTI) ratio to guess the % chance they stop paying.
  * **LGD (Loss Given Default):** If they default, how much of our money is gone forever?
  * **EAD (Exposure at Default):** How much money did they actually borrow?

### Engine 3: Portfolio VaR (Market Risk)
* **The Goal:** Tell the bank exactly how much money they could lose in the stock market on a really bad day.
* **How it works:** It uses **Monte Carlo Simulations**. Think of this like Doctor Strange looking at 10,000 alternate futures. 
* **The Math:** Sentinel takes a portfolio of stocks (like Apple, Microsoft, and the S&P 500), looks at their historical daily returns, and uses random number generation to simulate 10,000 possible ways tomorrow could play out. It then finds the 95th percentile worst-case scenario. This number is called the **VaR (Value at Risk)**.

### Engine 4: Volatility Surface
* **The Goal:** Understand if the stock market is currently calm or panicked.
* **How it works:** Stock market volatility isn't static; it comes in "bursts." A quiet market usually stays quiet, and a crashing market usually stays chaotic for a while.
* **The Math:** Standard deviation (basic math) assumes volatility is always the same. Sentinel uses an advanced model called **GARCH(1,1)**. GARCH specifically measures the "memory" of the market to calculate exactly how persistent the current panic or calmness will be over the long run.

### Engine 5: Scenario Analysis (Stress Testing)
* **The Goal:** Simulate what would happen to the bank if a specific historical catastrophe happened again today.
* **How it works:** Sentinel allows the user to say, "What if the 2008 Financial Crisis happens tomorrow?"
* **The Math:** It takes the exact mathematical shocks from historical crises (e.g., Tech stocks dropping 30%, real estate dropping 20%) and instantly applies them to the current live portfolio to see if the bank would survive the hit.

---

## 4. The Capstone: Obligor 360

In the banking world, an "Obligor" is just a fancy word for a customer. 

The biggest problem in traditional banks is that the Fraud Team, the Credit Team, and the Market Team don't talk to each other. A customer might look great to the Credit team, but the Fraud team knows they are acting suspiciously. 

**Obligor 360** fixes this. You can type in a single Customer ID (e.g., `B-0001`), and Sentinel instantly queries all of the databases at the exact same time. It mathematically combines their Credit Probability of Default with their active Fraud Flags to generate a single, unified **Risk Exposure Dollar Amount**. 

It gives the bank a perfect, 360-degree view of the danger a single human poses to the institution.

---

## 5. How to Read This Codebase

If you want to read the code, here is where to look:

* `sentinel/api/main.py`: This is the traffic cop. It handles all the web requests, buttons, and HTML generation.
* `sentinel/quant/`: This folder contains all the heavy math. (Monte Carlo, GARCH, Stress testing).
* `sentinel/fraud/` & `sentinel/credit/`: This is where the Machine Learning (AI) happens.
* `sentinel/data/generator.py`: This is the simulation engine that generates the fake bank data.
