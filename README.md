# Knowing When Your LLM Is Wrong

Companion code for the blog post "Knowing When Your LLM Is Wrong."

A single, self-contained script that walks through the full evaluation
pipeline for a binary routing agent:

    1. Build a labeled gold set
    2. Run the agent and compute error rate + confusion matrix
    3. Estimate the Bayes floor from inter-annotator agreement
    4. Extract a confidence signal via self-consistency sampling
    5. Calibrate the signal with Platt scaling
    6. Measure calibration with ECE and a reliability diagram
    7. Tune a decision threshold (with an abstention zone)
    8. Run an A/B test between two policies, properly

The "LLM" here is a mock function so the script runs offline with no
API key. The structure is identical to what you'd write against a real
commercial API: replace `MockLLM.classify` with a real API call and the
rest of the pipeline is unchanged.

Dependencies: `numpy, scikit-learn, matplotlib`
    ```pip install numpy scikit-learn matplotlib```

Run:
    ```python llm_correctness_demo.py```
