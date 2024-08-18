from flask import Flask, request, render_template
from tensorflow.keras.models import load_model  #type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img #type: ignore
import numpy as np
import os
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import uuid
import json
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('crop_tuned_model.keras')

train_path = r'plant/train' 
class_labels = sorted(os.listdir(train_path))

accuracy_rate = 99.01

def plot_prediction(pred_percentages):
    sorted_dict=sorted(pred_percentages.items(),key=lambda x:x[1],reverse=True)
    labels,values=zip(*sorted_dict)
    fig=go.Figure(data=[
    go.Bar(x=labels,y=values,
    marker_color='rgba(75, 192, 192, 0.6)',
    hoverinfo='x+y',
    textposition='auto'
               )
    ])
    fig.update_layout(
        title='Crop Disease Prediction Probabilities',
        xaxis_title='Disease Types',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        height=600,
        width=1200,
    )

    plot_filename = f"plot_{uuid.uuid4().hex}.png"
    plot_path = os.path.join('static', 'plots', plot_filename)
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.write_image(plot_path)
    plot_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    return plot_filename,plot_json

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image = load_img(filepath, target_size=(256, 256))  
            image = img_to_array(image) / 255.0 
            image = np.expand_dims(image, axis=0)
            predictions = model.predict(image)
            label_index = np.argmax(predictions, axis=1)[0]
            label = class_labels[label_index]
            confidence = float(np.max(predictions)) 
            prediction_percentages = {class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))}
            plot_file,plot_json=plot_prediction(prediction_percentages)
            prediction={
                    'label': label,
                    'confidence': confidence,
                    'accuracy_rate': accuracy_rate,
                }
            return render_template('output.html',prediction=prediction,plot_file=plot_file,plot_json=plot_json)
        except Exception as e:
            return str(e)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
