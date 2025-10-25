from flask import Flask, render_template, request, jsonify
import youtube_downloader

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/process-videos', methods=['POST'])
def process_videos():
    data = request.get_json()
    urls = data.get('urls', [])
    
    # Process the YouTube URLs
    results = []
    for url in urls:
        # Here you can add your processing logic
        # For now, we'll just validate and return the URLs
        frame_skip = 30
        if 'youtube.com' in url or 'youtu.be' in url:
            analyzer = youtube_downloader.YouTubeFaceEmotionAnalyzer(output_dir="emotion_faces")
            result = analyzer.process_single_video(url, frame_skip=frame_skip)
            results.append({
                'url': url,
                'status': 'valid',
                'result': result,
                'message': 'YouTube URL received successfully'
            })
        else:
            results.append({
                'url': url,
                'status': 'invalid',
                'result': '',
                'message': 'Not a valid YouTube URL'
            })
    
    return jsonify({
        'success': True,
        'total': len(urls),
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True)