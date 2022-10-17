from flask import Flask, request, render_template, g
from search import search
import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.before_request
def before_request():
    g.request_start_time = time.time()
    g.request_time = lambda: "%.5fs" % (time.time() - g.request_start_time)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['get'])
def process():
    mode = request.args.get('mode')
    query = request.args.get('query')

    results = search(query, mode)
    return render_template('search.html', results=results, time=g.request_time())


if __name__ == '__main__':
    app.run()
