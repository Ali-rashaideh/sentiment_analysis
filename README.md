# Sentiment Analysis API

This project provides a FastAPI-based REST API for sentiment analysis in English and Arabic, using locally stored HuggingFace models. It supports single and batch text prediction, combining results from language-specific and multilingual models for robust sentiment classification.

## Features
- Sentiment analysis for English and Arabic texts
- Batch prediction support
- Combines predictions from primary and multilingual models
- Returns confidence scores and model details

## Setup

1. **Clone the repository**
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

## File Structure
- `api.py`: FastAPI server
- `models.py`: Sentiment analysis logic
- `downloader.py`: Model downloader
- `requirements.txt`: Python dependencies
- `models/`: Directory containing downloaded models

## Notes
- Ensure you have enough disk space and memory for model downloads and inference.
- The API supports both English and Arabic texts, automatically selecting the appropriate model.
