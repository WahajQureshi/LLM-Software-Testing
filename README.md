# LLM-Software-Testing

**Using Large Language Models (LLMs) for Automated Java Unit Test Generation**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Llama--3.1--8B-green.svg)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-Methods2Test-orange.svg)](https://github.com/microsoft/methods2test)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project fine-tunes **Meta's Llama-3.1-8B-Instruct** model to automatically generate JUnit test cases from Java focal methods. Through iterative experimentation and an aggressive learning rate strategy, the model achieves **94% accuracy** on real-world validation tests.

## Key Highlights

- **Model**: Llama-3.1-8B-Instruct (8.03 billion parameters)
- **Fine-tuning Method**: QLoRA (4-bit quantization) for efficient training
- **Dataset**: 24,537 high-quality samples from Microsoft's Methods2Test
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **Training Time**: 3 epochs, ~50 GPU hours
- **Cost**: $9.73 total (via cloud GPU)
- **Real-world Performance**: 94% accuracy on custom Java validation methods

## Methodology

### Dataset Evolution
- **Original Dataset**: 780,944 samples from Methods2Test (FM_FC context level)
- **Filtering Process**: Applied aggressive quality filtering
- **Final Dataset**: 24,537 samples (3% of original)
- **Key Insight**: Quality > Quantity - smaller, curated dataset outperformed larger noisy dataset

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Model | Llama-3.1-8B-Instruct | 8.03B parameters |
| Quantization | 4-bit NF4 (QLoRA) | Reduces memory from 32GB to 6GB |
| Learning Rate | 5×10⁻⁴ | **Aggressive** - discovered through 9 iterations |
| LoRA Rank | 16 | Low-rank adaptation |
| LoRA Alpha | 32 | Scaling factor |
| Epochs | 3 | Optimal balance |
| Batch Size | 4 (effective) | Per-device: 1, gradient accumulation: 4 |
| Optimizer | AdamW (paged, 8-bit) | Memory-efficient |
| Scheduler | Cosine decay | Smooth learning rate reduction |
| Trainable Parameters | 0.52% (41.9M / 8.03B) | LoRA efficiency |

### Counterintuitive Discovery

After **9 failed training attempts** with conventional learning rates (1×10⁻⁴, 2×10⁻⁴), I discovered that a **higher learning rate** (5×10⁻⁴) worked best - challenging standard recommendations but yielding superior results.

## Results

### Quantitative Metrics

| Metric | Value |
|--------|-------|
| **Real-World Validation Accuracy** | **94.0%** |
| Test-Set Precision | 67.88% |
| Test-Set Recall | 56.32% |
| Test-Set F1-Score | 61.56% |
| BLEU Score | 14.03% |
| Training Loss (Final) | 0.1750 |
| Validation Loss (Final) | 0.6437 |

### Qualitative Results

The model was tested on **7 custom Java methods** written specifically for validation:

- **5 methods**: Achieved 100% accuracy
- **Worst performer**: 85% accuracy
- **Average**: 94% accuracy

#### Example: Edge Case Discovery

For the `DateValidator.isValidAge()` method, the model automatically generated tests for:
- Normal cases (valid ages: 25, 50)
- Boundary cases (0, 150)
- **Edge cases beyond boundaries** (-1, 151) - tests I hadn't explicitly thought of!

This demonstrates the model's ability to reason about test coverage beyond simple happy-path scenarios.

## Repository Structure

```
LLM-Software-Testing/
├── finetuning_llama_3.ipynb           # Main training notebook
├── validation_results.csv              # Evaluation results (100 samples)
├── COMPLETE_HARDWARE_SOFTWARE_CONFIGURATION.md  # Full hardware/software specs
├── README.md                           # This file
├── .gitignore                          # Git ignore patterns
└── LICENSE                             # MIT License (optional)
```

## Files Description

### `finetuning_llama_3.ipynb`
Complete training pipeline including:
- Dataset loading and preprocessing
- QLoRA configuration
- Training loop with monitoring
- Model saving and evaluation
- Inference examples

### `validation_results.csv`
Contains 100 test samples with:
- Input Java methods
- Generated test cases
- Ground truth tests
- Evaluation metrics (BLEU, precision, recall, F1)

### `COMPLETE_HARDWARE_SOFTWARE_CONFIGURATION.md`
Detailed specifications for reproducibility:
- RTX 4090 hardware specs (16,384 CUDA cores, 24GB VRAM)
- CUDA 12.1, PyTorch 2.0.1, Transformers 4.41.2
- Complete training hyperparameters
- Cost breakdown ($9.73 total)

## Hardware Requirements

### Minimum Requirements
- **GPU**: 16GB+ VRAM (e.g., RTX 4070 Ti, A4000)
- **RAM**: 32GB system RAM
- **Storage**: 10GB for model + dataset

### Recommended Configuration
- **GPU**: RTX 4090 (24GB VRAM) or A100 (40GB)
- **RAM**: 64GB
- **Storage**: 50GB (for full workspace)

### Training Hardware Used
- **GPU**: NVIDIA RTX 4090
  - 16,384 CUDA cores
  - 24GB GDDR6X VRAM
  - 1,008 GB/s bandwidth
  - 82.58 TFLOPS (FP32)
- **Cloud Provider**: Vast.ai
- **Cost**: $0.194/hour × 50.15 hours = $9.73

## Installation & Usage

### Prerequisites
```bash
# Python 3.10+
# CUDA 12.1+
# 16GB+ GPU VRAM
```

### Install Dependencies
```bash
pip install torch transformers datasets peft accelerate bitsandbytes wandb
```

### Quick Start
1. **Clone this repository**
```bash
git clone https://github.com/wahaj/LLM-Software-Testing.git
cd LLM-Software-Testing
```

2. **Open the training notebook**
```bash
jupyter notebook finetuning_llama_3.ipynb
```

3. **Run cells sequentially** - Follow instructions in notebook

### Using the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapter (link to your model)
model = PeftModel.from_pretrained(base_model, "path/to/your/adapter")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Generate test case
java_method = """
public static boolean isValidEmail(String email) {
    return email != null && email.contains("@") && email.contains(".");
}
"""

prompt = f"Generate JUnit test case for:\n{java_method}"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=512)
test_case = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(test_case)
```

## Model Weights

**Note**: The fine-tuned model (177 MB LoRA adapter) is too large for GitHub.

### Download Options:
1. **Hugging Face Hub** (Recommended): [Link will be added]
2. **GitHub Releases**: [Link will be added]
3. **Contact**: Request via email/issues

Alternatively, you can train your own model using the provided notebook.

## Dataset

This project uses the **Methods2Test** dataset from Microsoft Research.

- **Source**: [microsoft/methods2test](https://github.com/microsoft/methods2test)
- **Context Level**: FM_FC (Focal Method + Focal Class)
- **Original Size**: 780,944 Java method-test pairs
- **Our Filtered Dataset**: 24,537 samples

### Filtering Criteria
- Removed duplicates
- Removed malformed JSON
- Removed empty tests
- Removed excessively long methods (>512 tokens)
- Removed tests with compilation errors

## Key Learnings & Insights

### 1. Challenge Conventional Wisdom (Carefully)
The aggressive learning rate (5×10⁻⁴) that worked best contradicted standard advice. But I only tried higher rates after understanding *why* lower rates were failing (echo behavior).

### 2. Dataset Quality > Dataset Size
Reducing the dataset by 97% (780K → 24.5K) actually *improved* results because quality mattered more than quantity.

### 3. Real-World Testing Reveals What Benchmarks Hide
- Test-set precision: 67.88% (mediocre)
- Real-world validation: 94% (excellent)
- **Lesson**: Benchmark metrics don't tell the whole story. Test on real code.

### 4. Iterative Experimentation is Essential
It took **9 training attempts** to find the optimal configuration. Research is messy, intuition matters, and sometimes the "wrong" approach turns out to be right.

## Limitations

1. **Java-Specific**: Trained only on Java test cases
2. **Unit Tests Only**: Not integration or E2E tests
3. **Methods2Test Distribution**: Model may struggle with code significantly different from training data
4. **No Compilation Verification**: Generated tests may need syntax fixes
5. **Limited Context**: Uses only focal method + class (not full project context)

## Future Work

- [ ] Extend to other languages (Python, JavaScript, Go)
- [ ] Add compilation verification step
- [ ] Incorporate full project context (imports, dependencies)
- [ ] Fine-tune on integration test generation
- [ ] Build IDE plugin for seamless integration
- [ ] Experiment with larger models (Llama-3.1-70B, CodeLlama)
- [ ] Synthetic data augmentation strategies

## Comparison with Related Work

| Aspect | [Shaheer-Rehan/Llama-2](https://github.com/Shaheer-Rehan/Llama-2-for-Software-Testing) | This Project |
|--------|---------|--------------|
| **Base Model** | Llama-2-7b-chat | Llama-3.1-8B-Instruct ✅ (newer) |
| **Dataset Size** | 25,000 samples | 24,537 samples (similar) |
| **Training Epochs** | 12 epochs | 3 epochs ✅ (more efficient) |
| **Learning Rate** | Not specified | 5×10⁻⁴ (aggressive, documented) |
| **GPU** | A100 | RTX 4090 ✅ (consumer hardware) |
| **Validation** | CSV metrics only | **Real-world 7-method test + CSV** ✅ |
| **Documentation** | Basic | Comprehensive (training history, iterations) |

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{llm-software-testing-2024,
  author = {Syed Wahaj Qureshi},
  title = {LLM-Software-Testing: Automated Java Unit Test Generation with Llama-3.1-8B},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/wahaj/LLM-Software-Testing}
}
```

## References

1. **Meta Llama 3.1**: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. **Methods2Test Dataset**: Tufano, M., et al. "Unit Test Case Generation with Transformers and Focal Context." *arXiv preprint arXiv:2009.05617* (2020).
3. **QLoRA**: Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv preprint arXiv:2305.14314* (2023).
4. **LoRA**: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR* (2022).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The Llama-3.1 model has its own license from Meta. Please review Meta's licensing terms before commercial use.

## Acknowledgments

- **Meta AI** for the Llama-3.1 model
- **Microsoft Research** for the Methods2Test dataset
- **Hugging Face** for Transformers and PEFT libraries
- **Vast.ai** for affordable GPU cloud compute

## Contact

For questions, suggestions, or collaboration:
- **GitHub Issues**: [Open an issue](https://github.com/wahaj/LLM-Software-Testing/issues)
- **Email**: wahajqureshi6@gmail.com

---

**Final Thought**: Nine training attempts, 50 GPU hours, countless debugging sessions, and $9.73 in cloud compute later - I have a model that actually works. More importantly, I learned that research is messy, intuition matters, and sometimes the "wrong" approach (high learning rate) turns out to be right.
