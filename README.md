# Sentiment Analysis API

This project provides a FastAPI-based REST API for sentiment analysis in English and Arabic, using locally stored HuggingFace models. It supports single and batch text prediction, combining results from language-specific and multilingual models for robust sentiment classification.

## Features
- Sentiment analysis for English and Arabic texts
- Batch prediction support
- Combines predictions from primary and multilingual models
- Returns confidence scores and model details

## Design Choices & Rationale

### Model Selection
- **Arabic:** [CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment] was chosen for its strong performance on Arabic sentiment tasks and its dedicated training on Arabic data, ensuring high accuracy and robust handling of Arabic text.
- **English:** [mrm8488/deberta-v3-small-finetuned-sst2] is a compact, fast, and accurate model for English sentiment analysis, fine-tuned on SST-2, making it suitable for real-time inference and resource-constrained environments.
- **Multilingual:** [cardiffnlp/twitter-xlm-roberta-base-sentiment] provides coverage for both languages and acts as a secondary vote, improving robustness and handling edge cases or mixed-language input.

These models were selected for their balance of inference speed, size, and language-specific performance. No custom training or fine-tuning was performed, as open-source models are sufficient for this use case.

### Frameworks & Libraries
- **Transformers (HuggingFace):** Industry-standard for NLP, supports easy local model loading, fast inference, and a wide range of pre-trained models.
- **FastAPI:** Chosen for its speed, simplicity, and automatic OpenAPI documentation, making it ideal for production-ready REST APIs.
- **PyTorch:** Used as the backend for model inference, providing efficient computation and compatibility with HuggingFace models.

### Hardware Considerations
- **CPU:** The selected models are small enough to run efficiently on modern CPUs for moderate workloads. For batch or high-throughput scenarios, a CPU with at least 4 cores and 16GB RAM is recommended.


### LLM Choice Rationale
- **Inference Speed:** All selected models are optimized for fast inference, with small parameter sizes and efficient architectures.
- **Size:** Models are small enough to fit in memory on most consumer hardware, avoiding the need for expensive infrastructure.
- **Arabic Performance:** The CAMeL-Lab model is specifically designed for Arabic, outperforming generic multilingual models on Arabic sentiment tasks.

### Summary
This solution prioritizes:
- High accuracy for both English and Arabic
- Fast inference and low resource requirements
- Robustness via model ensembling
- Easy deployment on both CPU and GPU

## Setup

1. **Clone the repository**
   ```powershell
   git clone https://github.com/Ali-rashaideh/sentiment_analysis.git
   ```
2. **Install dependencies**
   
   ```powershell
   pip install -r requirements.txt
   ```

3. **Download models**
   
   ```powershell
   python downloader.py
   ```

4. **Run the API server**
   
   ```powershell
   uvicorn api:app --reload
   ```

## API Endpoints

### Health Check
- `GET /health`
  - Returns `{ "ok": true }` if the server is running.

### Predict Sentiment
- `POST /predict`
  - Request body (JSON):
    - `text`: Single string for analysis
    - `texts`: List of strings for batch analysis
  - Response: List of sentiment analysis results

## Example Request

Use the following `curl` command to test batch sentiment prediction:

```bash
curl --location 'http://127.0.0.1:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "texts": [
        "The customer was experiencing an issue with the laptop that has not been resolved yet after several attempts, and an appointment was scheduled to follow up on the case next Sunday",
        "تواصل العميل مع مركز خدمه العملاء للحصول على شهاده رصيد حسابها. تم توجيهها الى الموقع الالكتروني للبنك لاصدار الشهاده وتوجيهها الى جهه معنيه",
        "كان العميل يواجه صعوبة في سماع الوكيل أثناء المكالمة بسبب انخفاض مستوى الصوت. وافق الوكيل على إرسال رسالة نصية عبر تطبيق واتساب وأبدى العميل امتنانه بينما ينتظر اتصالاً معاوداً"
    ]
}'
```

## Output
The response will be a JSON object containing the sentiment analysis results for each input text, including the predicted label, confidence, and model details.
### Example Output:
```bash
        {
            "sentence": "كان العميل يواجه صعوبة في سماع الوكيل أثناء المكالمة بسبب انخفاض مستوى الصوت. وافق الوكيل على إرسال رسالة نصية عبر تطبيق واتساب وأبدى العميل امتنانه بينما ينتظر اتصالاً معاوداً",
            "final_label": "positive",
            "final_confidence": 0.5941414616709052,
            "final_margin": 0.37070129738553703,
            "vote_weights": {
                "primary": 0.6,
                "secondary": 0.4
            },
            "used_models": [
                "arabic",
                "multi"
            ],
            "best_model": "arabic",
            "best_model_confidence": 0.9577555972435603,
            "per_model": [
                {
                    "model": "arabic",
                    "label": "positive",
                    "scores": {
                        "positive": 0.9577555972435603,
                        "natural": 0.039879929346705575,
                        "negative": 0.002364473409734113
                    }
                },
                {
                    "model": "multi",
                    "label": "negative",
                    "scores": {
                        "positive": 0.048720258311922504,
                        "natural": 0.3962260410892583,
                        "negative": 0.5550537005988192
                    }
                }
            ]
        }
    ]
}
```

## File Structure
- `api.py`: FastAPI server
- `models.py`: Sentiment analysis logic
- `downloader.py`: Model downloader
- `requirements.txt`: Python dependencies
- `models/`: Directory containing downloaded models

## Notes
- Ensure you have enough disk space and memory for model downloads and inference.
- The API supports both English and Arabic texts, automatically selecting the appropriate model.
