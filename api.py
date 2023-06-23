from app import vc_fn
import asyncio
import traceback
from datetime import datetime
import numpy as np
from flask import Flask, request
from flask_cloudflared import run_with_cloudflared

app = Flask(__name__)
run_with_cloudflared(app)  # Open a Cloudflare Tunnel when app is run

@app.route('/api/vc', methods=['POST'])
def vc_api():
    # Retrieve the data from the POST request
    data = request.get_json()

    # Call the vc_fn function with the provided data
    result = vc_fn(
        data['input_audio'],
        data['upload_audio'],
        data['upload_mode'],
        data['f0_up_key'],
        data['f0_method'],
        data['index_rate'],
        data['tts_mode'],
        data['tts_text'],
        data['tts_voice']
    )

    # Return the result as a JSON response
    return {
        'message': result[0],  # Assuming the first value in the result is the message
        'audio': result[1]  # Assuming the second value in the result is the audio
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
