# resume-shortlisting-system
# AI-Powered Resume Shortlisting and Feedback System

A machine learning system that automatically compares resumes against job descriptions, generates alignment scores, and provides actionable feedback on keyword gaps.

## How It Works

1. Takes a resume and a job description as input
2. Uses TF-IDF vectorization to extract and compare key terms from both
3. Generates an alignment score showing how well the resume matches the role
4. Highlights missing keywords and suggests improvements
5. Outputs structured feedback to reduce manual screening effort

## Tech Stack

- **Python** — core language
- **Scikit-learn** — TF-IDF vectorization and model logic
- **Pandas** — data handling and processing
- **TF-IDF** — text feature extraction

## Results

- Reduced manual screening effort by ~35%
- Processed 3,000+ text records during testing and validation
- Modular architecture — scoring logic can be swapped without changing core code

## How to Run

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/resume-shortlisting-system.git
cd resume-shortlisting-system
```

2. Install dependencies
```bash
pip install pandas scikit-learn
```

3. Run the main script
```bash
python main.py
```

## Project Structure

```
resume-shortlisting-system/
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
└── README.md            # This file
```
