from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import os
from dotenv import load_dotenv

# 載入 .env 變數
load_dotenv()
OCR_API_KEY = os.environ.get("OCR_API_KEY")
if not OCR_API_KEY:
    print("⚠️ 未設定 OCR_API_KEY，圖片辨識功能將無法使用")

app = Flask(__name__)
CORS(app)

# 載入模型與向量器
try:
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"❌ 模型或向量器載入失敗：{e}")
    model = None
    vectorizer = None

# 文字預測 API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': '請提供 text 欄位'}), 400

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        return jsonify({'label': 'spam' if pred == 1 else 'ham'})

    except Exception as e:
        print(f"❌ 預測錯誤：{e}")
        return jsonify({'error': str(e)}), 500

# 圖片 OCR + 預測 API
@app.route('/analyze-all', methods=['POST'])
def analyze_all():
    try:
        image_file = request.files.get('image', None)
        text_input = request.form.get('text', '').strip()
        extracted_text = ''

        # 有圖片就 OCR
        if image_file:
            if not OCR_API_KEY:
                return jsonify({'error': 'OCR_API_KEY_MISSING'}), 500

            # 判斷副檔名與 MIME 類型
            ext = os.path.splitext(image_file.filename)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                mime = 'image/jpeg'
            elif ext == '.png':
                mime = 'image/png'
            elif ext == '.bmp':
                mime = 'image/bmp'
            elif ext == '.gif':
                mime = 'image/gif'
            else:
                ext = '.jpg'
                mime = 'image/jpeg'

            ocr_response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': ('image' + ext, image_file, mime)},
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'cht'
                }
            )

            result = ocr_response.json()
            if not result.get('IsErroredOnProcessing'):
                extracted_text = result['ParsedResults'][0].get('ParsedText', '')
            else:
                details = result.get('ErrorMessage') or result.get('ErrorDetails') or 'unknown'
                return jsonify({'error': 'OCR_API_ERROR', 'details': details}), 500

        # 合併圖像 + 手動文字
        full_text = f"{extracted_text.strip()} {text_input}".strip()
        if not full_text:
            return jsonify({'error': '未提供有效文字'}), 400

        vec = vectorizer.transform([full_text])
        pred = model.predict(vec)[0]
        score = model.predict_proba(vec)[0][1]

        return jsonify({
            'final_label': 'spam' if pred == 1 else 'ham',
            'text': full_text,
            'total_score': round(score, 4)
        })

    except Exception as e:
        print(f"❌ 分析時錯誤：{e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
