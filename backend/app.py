from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  


model = YOLO('best.pt') 


fertilizer_dict = {
    'blister_blight': {
        'sinhala': 'කොපර් ඔක්සික්ලෝරයිඩ් හෝ මැන්කොසෙබ් භාවිතා කරන්න.',
        'msg': 'Use Copper oxychloride or Mancozeb fungicide.',
        'organic': 'නීම් තෙල් (Neem Oil) ස්ප්‍රේ කරන්න.',
        'non_organic': 'Copper based fungicides'
    },
    'gray_blight': {
        'sinhala': 'කැබෙන්ඩාසින් හෝ හෙක්සාකොනාසෝල් භාවිතා කරන්න.',
        'msg': 'Use Carbendazim or Hexaconazole.',
        'organic': 'සුදුළූණු සහ නීම් සාරය (Garlic + Neem) ස්ප්‍රේ කරන්න.',
        'non_organic': 'Carbendazim'
    },
    'brown_blight': {
        'sinhala': 'මැන්කොසෙබ් හෝ ක්ලොරොතැලෝනිල් භාවිතා කරන්න.',
        'msg': 'Use Mancozeb or Chlorothalonil.',
        'organic': 'බේකින් සෝඩා මිශ්‍ර ජලය ස්ප්‍රේ කරන්න.',
        'non_organic': 'Mancozeb'
    },
    'red_rust': {
        'sinhala': 'කොපර් හයිඩ්‍රොක්සයිඩ් භාවිතා කරන්න.',
        'msg': 'Use Copper hydroxide.',
        'organic': 'සබන් මිශ්‍ර නීම් තෙල් ස්ප්‍රේ කරන්න.',
        'non_organic': 'Copper hydroxide'
    },
    'healthy': {
        'sinhala': 'ශාකය නිරෝගීයි. දිගටම හොඳින් නඩත්තු කරන්න.',
        'msg': 'Plant is healthy. Keep good maintenance.',
        'organic': 'කොම්පෝස්ට් පොහොර යොදන්න.',
        'non_organic': 'NPK සමබර පොහොර යොදන්න'
    }
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
       
        image = Image.open(io.BytesIO(file.read()))
        
        
        results = model(image)
        
        
        detected_disease = "healthy"
        highest_conf = 0
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                
                if conf > 0.50 and conf > highest_conf:
                    highest_conf = conf
                    cls_id = int(box.cls[0])
                    detected_disease = model.names[cls_id]
        
        
        info = fertilizer_dict.get(detected_disease, fertilizer_dict['healthy'])
        
        
        return jsonify({
            'disease_name': detected_disease,
            'confidence': round(highest_conf * 100, 2), 
            'sinhala_advice': info['sinhala'],
            'english_advice': info['msg'],
            'organic_remedy': info.get('organic', '-'),
            'chemical_remedy': info.get('non_organic', '-')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
