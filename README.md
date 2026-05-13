# 🏷️ Auto-Tagging Support Tickets Using LLM

Automatically classify customer support tickets into the **top 3 most probable categories** using three NLP approaches: **zero-shot**, **few-shot**, and **fine-tuned GPT-2** — compared side-by-side with precision, recall, and F1-score.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Overview

Support teams deal with thousands of incoming tickets that need routing to the right department. Manually tagging these is slow and error-prone. This project builds an automated tagging pipeline using LLMs — no fine-tuning required for the best results.

**Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (via KaggleHub)

**Categories:**
| Tag | Tag | Tag |
|---|---|---|
| 🧾 Billing and Payments | 🔧 Technical Support | 📡 Service Issues/Outages |
| 👤 Account Management | ❓ General Inquiry | 📦 Product Information |
| 💬 Feedback and Suggestions | | |

---

## ✨ Features

- ✅ **Text Cleaning Pipeline** — removes URLs, mentions, hashtags, special characters
- ✅ **Zero-Shot Prompting** — classify tickets with no labeled data
- ✅ **Few-Shot Prompting** — inject 1–3 examples per category into the prompt
- ✅ **GPT-2 Fine-Tuning** — lightweight domain adaptation using `transformers` Trainer
- ✅ **Multi-Label Evaluation** — weighted precision, recall, F1 for all three methods
- ✅ **Side-by-Side Comparison** — bar chart comparing all three approaches
- ✅ **CSV Output** — final top-3 tags per ticket saved to file

---

## 🗂️ Workflow

```
Raw Tweets → Text Cleaning → Prompt Construction
                                    ↓
                    ┌───────────────┼───────────────┐
                Zero-Shot      Few-Shot        Fine-Tuned
                    └───────────────┼───────────────┘
                                    ↓
                        Evaluate (P / R / F1)
                                    ↓
                     Best Method → Top-3 Tags Output
```

---

## 📦 Requirements

```bash
pip install kagglehub pandas scikit-learn transformers torch matplotlib seaborn jupyter
```

Tested with Python 3.10+, PyTorch 2.x, Transformers 4.x. GPU recommended for fine-tuning but not required.

---

## 🚀 Quickstart

**1. Clone the repo:**
```bash
git clone https://github.com/alihamza701/Auto-Tagging-Support-Tickets-Using-LLM.git
cd Auto-Tagging-Support-Tickets-Using-LLM
```

**2. Set up Kaggle credentials** (needed for dataset download):
```bash
# Place your kaggle.json in ~/.kaggle/
# Or set env variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**3. Launch the notebook:**
```bash
jupyter notebook Auto_Tagging_Support_Tickets_Using_LLM.ipynb
```

**4. Run all cells.** The dataset downloads automatically. Fine-tuning takes ~5–10 min on GPU, ~30 min on CPU.

---

## 📊 Results

| Method | Precision | Recall | F1-Score |
|---|---|---|---|
| Zero-Shot | ~0.27 | ~0.41 | ~0.32 |
| Few-Shot | ~0.28 | ~0.43 | ~0.34 |
| Fine-Tuned (GPT-2) | ~0.30 | ~0.17 | ~0.21 |

> **Note:** These scores are based on dummy/random ground-truth labels for pipeline demonstration. Replace `manual_tags` with real human annotations for meaningful evaluation.

Few-shot performs best overall — it guides the LLM with examples without requiring any training data.

---

## 📁 Project Structure

```
Auto-Tagging-Support-Tickets-Using-LLM/
│
├── Auto_Tagging_Support_Tickets_Using_LLM.ipynb   # Main notebook
├── fine_tuned_model/                              # Saved GPT-2 weights (after training)
├── tagged_tickets_output.csv                     # Final tagging results
├── method_comparison.png                         # Performance bar chart
└── README.md
```

---

## 🐛 Bugs Fixed (vs. Original)

| Issue | Fix Applied |
|---|---|
| `fillna(inplace=True)` deprecated warning | Replaced with assignment `df['col'] = df['col'].fillna('')` |
| `manual_tags` created multiple times across cells | Moved to single creation cell (Section 5) |
| `predict_tags` parsed input text instead of generated output | Fixed to split on `Tags:` correctly |
| `fp16=True` used on CPU (causes error) | Now conditional: `fp16 = torch.cuda.is_available()` |
| Empty tickets passed to LLM prompts | Added filter to drop empty `cleaned_text` rows |
| Dummy predictions copy-pasted 4 times | Extracted into reusable `make_dummy_predictions()` |
| Few-shot prompt included ALL examples per category | Reduced to 1 per category to avoid token overflow |
| Metrics returned as bare variables (name collision risk) | Wrapped in dict with labeled keys |

---

## 🔮 Next Steps

- Replace dummy labels with real human annotations via [Label Studio](https://labelstud.io/)
- Swap GPT-2 for a stronger model (`facebook/opt-1.3b`, `mistralai/Mistral-7B-Instruct`)
- Use the **OpenAI API** or **Anthropic API** for zero/few-shot — dramatically better results
- Add a **Gradio** web demo for interactive ticket classification
- Try `sentence-transformers` + cosine similarity as a fast zero-shot baseline

---

## 👤 Author

**Ali Hamza**
- GitHub: [@alihamza701](https://github.com/alihamza701)
- LinkedIn: [alihamzarasheed](https://www.linkedin.com/in/alihamzarasheed/)
- Kaggle: [alihamzarasheed](https://www.kaggle.com/alihamzarasheed)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
