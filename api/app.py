import json, os, sys, pickle
from flask import Flask, jsonify, request, abort
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

app = Flask(__name__)

SUMMARY_PATH  = os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'forecast_summary.json')
MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'trained_models')


def load_summary():
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {}

FORECAST_DATA = load_summary()


def state_key(state_raw: str) -> str:
    """Case-insensitive state lookup."""
    for k in FORECAST_DATA:
        if k.lower() == state_raw.lower():
            return k
    return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'states_loaded': len(FORECAST_DATA),
        'service': 'Sales Forecasting API v1.0'
    })


@app.route('/states', methods=['GET'])
def list_states():
    return jsonify({
        'states': sorted(FORECAST_DATA.keys()),
        'count':  len(FORECAST_DATA)
    })


@app.route('/forecast/<state_name>', methods=['GET'])
def get_forecast(state_name):
    """
    Returns 8-week sales forecast for a given state.
    Optional query param: ?model=SARIMA|Prophet|XGBoost|LSTM
    """
    key = state_key(state_name)
    if key is None:
        abort(404, description=f"State '{state_name}' not found. "
                               f"Use /states to see available states.")

    data       = FORECAST_DATA[key]
    model_req  = request.args.get('model', None)

    if model_req:
        model_req_norm = model_req.strip()
        if model_req_norm not in data.get('all_forecasts', {}):
            abort(400, description=f"Model '{model_req_norm}' not available for {key}. "
                                   f"Available: {list(data['all_forecasts'].keys())}")
        raw_forecast = data['all_forecasts'][model_req_norm]
        model_used   = model_req_norm
    else:
        raw_forecast = [row['predicted_sales'] for row in data['forecast']]
        model_used   = data['best_model']

    # Attach dates
    dates = [row['date'] for row in data['forecast']]
    forecast_out = [{'date': d, 'predicted_sales': round(v, 2)}
                    for d, v in zip(dates, raw_forecast)]

    return jsonify({
        'state':           key,
        'model_used':      model_used,
        'best_model':      data['best_model'],
        'forecast_periods': len(forecast_out),
        'forecast':        forecast_out,
        'metrics': data['metrics'].get(model_used, {})
    })


@app.route('/models/<state_name>', methods=['GET'])
def get_model_comparison(state_name):
    """Returns all model metrics for a state."""
    key = state_key(state_name)
    if key is None:
        abort(404, description=f"State '{state_name}' not found.")

    data = FORECAST_DATA[key]
    return jsonify({
        'state':      key,
        'best_model': data['best_model'],
        'metrics':    data['metrics'],
        'all_forecasts_available': list(data.get('all_forecasts', {}).keys())
    })


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger retraining. In production, this would be async (Celery/RQ).
    Here it reloads the precomputed summary from disk.
    """
    global FORECAST_DATA
    FORECAST_DATA = load_summary()
    return jsonify({
        'status':  'reloaded',
        'states':  len(FORECAST_DATA),
        'message': 'Forecast data reloaded from disk.'
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': str(e)}), 404

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Starting Forecasting API on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
