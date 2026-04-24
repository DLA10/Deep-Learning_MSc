# CI7521 - Machine Learning & Deep Learning - Coursework 2

## CRITICAL STYLE RULES (READ FIRST)

- **MINIMAL CODE**: Write the simplest, shortest, most straightforward code possible. No over-engineering, no unnecessary abstractions, no helper classes or utility functions unless absolutely needed. If something can be done in 3 lines, do not write 15.
- **HUMAN EXPLANATIONS**: All markdown explanations, discussions, and justifications must sound like a real student wrote them — natural, conversational, clear. No robotic language, no overly technical jargon, no buzzword-stuffed paragraphs. Explain things the way you'd explain them to a classmate. Keep it simple and honest.
- **NO COMPLICATED CONCEPTS**: Do not introduce advanced techniques, obscure optimisers, or niche methods unless the question specifically demands it. Stick to what would be taught in a standard MSc deep learning module.

## Instructions

- **Task**: Build a deep learning–based system to perform **multi-class classification** of medical images.
- **Submit**: ONE `.ipynb` notebook. KU numbers and names in the first cell. Write any explanations, discussion as comments, raw and/or markdown cells.
- **Stack**: Python, TensorFlow/Keras. Deep learning only — any solutions that use classic ML without deep learning will be rejected.
- **Dataset**: `organsmnist_224.npz` (~783MB), located in the same folder. From MedMNIST2D (OrganSMNIST, Abdominal CT, 224×224). Paper: https://www.nature.com/articles/s41597-022-01721-8 | Site: https://medmnist.com/ | GitHub: https://github.com/MedMNIST/MedMNIST
- **Template**: The notebook must be strictly based on the template provided on Canvas.
- **Libraries**: Free to use standard libs + tutor-provided sample code (with references). Usage of any other source code must be reported and referenced. Usage of any third-party libraries must be agreed (by email) with the Tutor beforehand. **Non-compliance may result in the project being rejected.**

### Tasks

**Before training any model, make sure** you perform Initial Visualisation to fully visualise the dataset — both with labels and without labels.

Using the training and validation subsets ONLY:
1. Train and validate a baseline NN of one hidden layer with 100 neurons.
2. Train, validate and discuss a new NN architecture of your choice that can outperform the one above.
3. Train and validate a baseline CNN of ONE Conv block 1.
4. Train, validate and discuss a new CNN architecture of your choice that can outperform the one above.
5. Choose two CNN architectures of your choice and apply them using Transfer learning and discuss.

In parts 2 and 4, you have to show 5 trials/models only and provide a summary simulation table showing the different results with different combinations of hyperparameters. Most probably, you will need to train/try out more than 5 models, in this case, do them separately on a draft notebook; you just need to show 5 on the main Notebook.

6. Once you are happy with the validation results, evaluate the performance of the SIX models previously built on the test dataset and report & discuss the results in a comparative study (example metrics: F1, precision, recall, accuracy, confusion matrix, etc.)
7. Next, select the best-performing model from the above 6 models. Apply data augmentation techniques to partially or fully address the imbalance in the dataset. Re-train the model and provide a comparative analysis and discussion of the performance of the model before and after data augmentation.

**Save models** after training so they can be reloaded for Q6 without retraining.

**SIX MODELS COUNT**: Q1(1) + Q2(1 best) + Q3(1) + Q4(1 best) + Q5(2 transfer learning architectures) = 6 total models for Q6 evaluation.

---

## FAQ

- **FAQ1**: Should any model be always better than the one in the previous question? **A1**: Only the models in Q2 and Q4 that should be better than the ones in Q1 and Q3, respectively.
- **FAQ2**: In Q6, we are asked to test the six models in Q1-5 on the held-out dataset, however, by then we will have lost connection and consequently we need to retrain the models all over again. **A2**: Once you are happy with any model in any question, save it so that you can load whenever you want.
- **FAQ3**: In Q1 and Q2, there is no "discuss" like in the other questions, is there any mistakes? **A3**: There is no mistake whatsoever, Q1 and Q3 are very straightforward and their purpose is to create a baseline for subsequent NN and CNN models, respectively, therefore, no discussion is required. **⚠️ NOTE**: The question says "Q1 and Q2" but the answer clarifies it is **Q1 and Q3** that need no discussion. Q2 DOES require discussion.
- **FAQ4**: What is meant by the "baseline CNN of ONE Conv block 1" in Q3? **A4**: It is a model that is composed of one simple Conv Block (One Conv2D Layer + one Maxpooling2D layer) and a head classifier part. The head classifier part could be as simple as (one Flatten layer + one Hidden layer + Output layer) or (One GAP layer + Output Layer).
- **FAQ5**: In Q5, do we choose the two CNN architectures randomly? Any potential list of such architectures? **A5**: It is up to you to choose any two CNN architectures as long as you support your choice. There are plenty of successful CNN architectures such as VGG16, ResNET50, AlexNet, etc.
- **FAQ6**: In the template, there is only one or two cells for each question, what if we need more? **A6**: You can add as many cells as you want.

---

## Grading (Top-Band Criteria)

| Section | Weight | Top-Band Requirement |
|---|---|---|
| Structure, Presentation & Initial Visualisation | 15% | Excellent structure, 100% template compliance, excellent visualisation |
| Q1 — Baseline NN | 5% | Excellent comprehensive code, clear training/validation results |
| Q2 — Improved NN | 15% | Excellent models, well-justified hyperparameters, clear results, strong discussion |
| Q3 — Baseline CNN | 5% | Excellent comprehensive code, clear training/validation results |
| Q4 — Improved CNN | 15% | Excellent models, well-justified hyperparameters, clear results, strong discussion |
| Q5 — Transfer Learning | 15% | Excellent architecture choices, training, validation, well-presented discussion |
| Q6 — Test Evaluation | 15% | Excellent hold-out testing, well-presented metrics and comparative discussion |
| Q7 — Data Augmentation | 15% | Excellent augmentation techniques, well-justified, strong before/after discussion |
