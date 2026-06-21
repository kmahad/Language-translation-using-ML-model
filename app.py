"""
Web Interface for SMT Machine Translation.

Provides API endpoints and visual control dashboard to run
data preparation, model training, evaluation, and translation.
"""

import os
import sys
import json
import yaml
import subprocess
import threading
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.data.tokenizer import load_tokenizers
from src.model import SMTModel
from src.evaluation import Evaluator


app = Flask(__name__)

# Cache for loaded model & evaluator
_cached_translator = None
_cached_checkpoint_path = None
_cached_mtime = 0

class ProcessRunner:
    """Manages execution of background machine learning tasks."""
    def __init__(self):
        self.process = None
        self.type = None  # 'prep', 'train', or 'test'
        self.log_file = None
        self.f_out = None
        self._lock = threading.RLock()

    def start(self, cmd, run_type, log_path):
        with self._lock:
            if self.is_running():
                return False
            self.type = run_type
            self.log_file = log_path
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # Reset log file
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"--- STARTING {run_type.upper()} PROCESS ---\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")

            self.f_out = open(log_path, 'a', encoding='utf-8')
            
            # Set PYTHONUNBUFFERED=1 to ensure outputs are flushed immediately
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            self.process = subprocess.Popen(
                cmd,
                stdout=self.f_out,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            return True

    def is_running(self):
        with self._lock:
            if self.process is None:
                return False
            status = self.process.poll()
            if status is not None:
                # Subprocess finished
                self.process = None
                if self.f_out:
                    try:
                        self.f_out.close()
                    except Exception:
                        pass
                    self.f_out = None
                return False
            return True

    def stop(self):
        with self._lock:
            proc = self.process
            if proc is None:
                return False
            
            # Check if it has already finished
            status = proc.poll()
            if status is not None:
                self.process = None
                if self.f_out:
                    try:
                        self.f_out.close()
                    except Exception:
                        pass
                    self.f_out = None
                return False
            
            # Attempt gentle terminate first
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if hung
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass
                
            self.process = None
            if self.f_out:
                try:
                    self.f_out.close()
                except Exception:
                    pass
                self.f_out = None
            return True

# Initialize global process runner
runner = ProcessRunner()

def get_dataset_path():
    """Resolves absolute path to CSV dataset file."""
    config = load_config('config/default.yaml')
    path = config.data.data_file
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    return path

def get_translator():
    """Loads and caches the trained translation model."""
    global _cached_translator, _cached_checkpoint_path, _cached_mtime
    
    config = load_config('config/default.yaml')
    checkpoint_path = os.path.join(config.training.checkpoint_dir, "best.json")
    
    if not os.path.exists(checkpoint_path):
        return None, "No trained checkpoint found. Please train the model first."

    current_mtime = os.path.getmtime(checkpoint_path)
    
    # Return cached if exists and checkpoint file has not changed
    if _cached_translator is not None and _cached_checkpoint_path == checkpoint_path and _cached_mtime == current_mtime:
        return _cached_translator, None

    try:
        # Load tokenizers
        tokenizers = load_tokenizers(config)
        
        # Build SMT model
        model = SMTModel(
            max_phrase_len=config.model.max_phrase_len,
            lm_order=config.model.lm_order,
            alignment_iterations=config.model.alignment_iterations,
        )
        model.load(checkpoint_path)
        
        evaluator = Evaluator(
            model=model,
            src_tokenizer=tokenizers["src"],
            tgt_tokenizer=tokenizers["tgt"],
            device=None,
            beam_size=config.inference.beam_size,
            max_decode_len=config.inference.max_decode_len,
        )
        
        _cached_translator = evaluator
        _cached_checkpoint_path = checkpoint_path
        _cached_mtime = current_mtime
        
        return evaluator, None
    except Exception as e:
        return None, f"Failed to load checkpoint: {str(e)}"

def parse_training_metrics():
    """Parses SMT tuning log file to get epochs and mock losses from BLEU score."""
    log_path = 'logs/web_train.log'
    if not os.path.exists(log_path):
        return []
    
    epochs_data = []
    import re
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            current_epoch = None
            best_bleu = 0.0
            for line in f:
                # Matches: Tuning Epoch 1/10
                epoch_match = re.search(r"Tuning Epoch (\d+)/\d+", line)
                if epoch_match:
                    if current_epoch is not None:
                        epochs_data.append({
                            "epoch": current_epoch,
                            "train_loss": max(0.0, 100.0 - best_bleu * 2),
                            "val_loss": max(0.0, 100.0 - best_bleu),
                            "lr": 0.0
                        })
                    current_epoch = int(epoch_match.group(1))
                
                # Matches: BLEU = 12.34
                bleu_match = re.search(r"BLEU = ([\d\.]+)", line)
                if bleu_match:
                    best_bleu = float(bleu_match.group(1))
                    
            if current_epoch is not None:
                epochs_data.append({
                    "epoch": current_epoch,
                    "train_loss": max(0.0, 100.0 - best_bleu * 2),
                    "val_loss": max(0.0, 100.0 - best_bleu),
                    "lr": 0.0
                })
    except Exception as e:
        print(f"Error parsing log metrics: {e}")
        
    return epochs_data

@app.route('/')
def home():
    """Renders the main single-page interface."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Checks translation system state and background process status."""
    config = load_config('config/default.yaml')
    dataset_path = get_dataset_path()
    checkpoint_path = os.path.join(config.training.checkpoint_dir, "best.json")
    
    # Tokenizer trained checks
    if config.tokenizer.shared_vocab:
        src_tok_path = os.path.join(config.tokenizer.model_dir, "sp_shared.model")
        tgt_tok_path = src_tok_path
    else:
        src_tok_path = os.path.join(config.tokenizer.model_dir, f"sp_{config.data.src_lang}.model")
        tgt_tok_path = os.path.join(config.tokenizer.model_dir, f"sp_{config.data.tgt_lang}.model")
    
    # Active device detection (SMT is CPU-only, no GPU needed)
    device_name = "CPU (SMT - No GPU needed)"

    # Model info from checkpoint
    checkpoint_info = None
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # Count translation rules in phrase table
            rules_count = sum(len(cand) for cand in checkpoint["phrase_table"]["phrase_probs"].values())
            
            checkpoint_info = {
                "epoch": "SMT Model",
                "best_val_loss": rules_count,  # displays in "Best Loss" area on UI
                "has_history": False
            }
        except Exception:
            checkpoint_info = {"epoch": "Error loading", "best_val_loss": 0.0}

    # Process metrics if training is active or logs exist
    metrics = parse_training_metrics()

    return jsonify({
        "running": runner.is_running(),
        "run_type": runner.type if runner.is_running() else None,
        "device": device_name,
        "dataset_exists": os.path.exists(dataset_path),
        "dataset_path": dataset_path,
        "tokenizer_exists": os.path.exists(src_tok_path) and os.path.exists(tgt_tok_path),
        "checkpoint_exists": os.path.exists(checkpoint_path),
        "checkpoint_info": checkpoint_info,
        "metrics": metrics,
        "config": {
            "src_lang": config.data.src_lang,
            "tgt_lang": config.data.tgt_lang,
            "vocab_size": config.tokenizer.vocab_size,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "d_model": "N/A (SMT)",
            "n_layers": f"MaxPhraseLen={config.model.max_phrase_len}",
            "device_type": "cpu"
        }
    })

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Fetches or updates the project's default.yaml configuration."""
    yaml_path = 'config/default.yaml'
    
    if request.method == 'GET':
        if not os.path.exists(yaml_path):
            return jsonify({"error": "Config file not found"}), 404
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return jsonify(config_dict)
    
    elif request.method == 'POST':
        try:
            updated_config = request.json
            if not updated_config:
                return jsonify({"error": "No config data received"}), 400
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(updated_config, f, default_flow_style=False)
                
            return jsonify({"status": "success", "message": "Config saved successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_task():
    """Launches a background translation pipeline task."""
    data = request.json or {}
    run_type = data.get('type')
    
    if run_type not in ('prep', 'train', 'test'):
        return jsonify({"error": "Invalid task type specified"}), 400
        
    if runner.is_running():
        return jsonify({"error": f"Task '{runner.type}' is already running"}), 400
        
    # Build command line execution list
    if run_type == 'prep':
        cmd = [sys.executable, 'scripts/prepare_data.py', '--config', 'config/default.yaml']
        log_path = 'logs/web_prep.log'
    elif run_type == 'train':
        cmd = [sys.executable, 'scripts/train.py', '--config', 'config/default.yaml']
        if data.get('smoke_test'):
            cmd.append('--smoke_test')
        log_path = 'logs/web_train.log'
    elif run_type == 'test':
        cmd = [sys.executable, 'scripts/test.py', '--config', 'config/default.yaml']
        if data.get('smoke_test'):
            cmd.append('--smoke_test')
        log_path = 'logs/web_test.log'
        
    success = runner.start(cmd, run_type, log_path)
    if success:
        return jsonify({"status": "success", "message": f"Task '{run_type}' started"})
    else:
        return jsonify({"error": "Failed to launch subprocess"}), 500

@app.route('/api/stop', methods=['POST'])
def stop_task():
    """Forces the running background subprocess to abort."""
    if not runner.is_running():
        return jsonify({"error": "No active tasks are currently running"}), 400
        
    run_type = runner.type
    success = runner.stop()
    if success:
        return jsonify({"status": "success", "message": f"Task '{run_type}' aborted successfully"})
    else:
        return jsonify({"error": "Failed to abort the subprocess"}), 500

@app.route('/api/logs/<run_type>', methods=['GET'])
def get_logs(run_type):
    """Streams terminal log lines from the background subprocess log file."""
    if run_type not in ('prep', 'train', 'test'):
        return jsonify({"error": "Invalid log type"}), 400
        
    log_path = f'logs/web_{run_type}.log'
    if not os.path.exists(log_path):
        return jsonify({"text": "", "offset": 0, "running": False})
        
    offset = int(request.args.get('offset', 0))
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(0, 2)
            total_size = f.tell()
            
            if offset > total_size:
                offset = 0 # reset if file truncated
                
            f.seek(offset)
            text = f.read()
            new_offset = f.tell()
            
        return jsonify({
            "text": text,
            "offset": new_offset,
            "running": runner.is_running() and runner.type == run_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset', methods=['GET'])
def get_dataset_page():
    """Reads raw CSV dataset rows using pandas with pagination."""
    path = get_dataset_path()
    if not os.path.exists(path):
        return jsonify({"error": f"Dataset file not found at: {path}"}), 404
        
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    
    try:
        config = load_config('config/default.yaml')
        sep = config.data.separator or ','
        
        # Read dataset
        df = pd.read_csv(path, sep=sep)
        total_rows = len(df)
        
        # Take page slice
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_df = df.iloc[start_idx:end_idx].fillna("")
        
        return jsonify({
            "columns": list(df.columns),
            "data": paginated_df.to_dict(orient='records'),
            "total_rows": total_rows,
            "page": page,
            "limit": limit
        })
    except Exception as e:
        return jsonify({"error": f"Error reading dataset: {str(e)}"}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Invokes cached model instance to translate input text."""
    data = request.json or {}
    text = data.get('text', '').strip()
    beam_size = data.get('beam_size')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    translator, error = get_translator()
    if error:
        return jsonify({"error": error}), 400
        
    try:
        # Override beam size if custom requested
        old_beam = translator.beam_size
        if beam_size is not None:
            translator.beam_size = int(beam_size)
            
        translation = translator.translate_sentence(text)
        
        # Restore beam size
        translator.beam_size = old_beam
        
        return jsonify({
            "source": text,
            "translation": translation,
            "beam_size": beam_size or old_beam
        })
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

@app.route('/api/eval_results', methods=['GET'])
def get_evaluation_results():
    """Retrieves generated translation hypotheses from the last test set run."""
    config = load_config('config/default.yaml')
    output_file = os.path.join(config.logging.log_dir, "translations.tsv")
    
    if not os.path.exists(output_file):
        return jsonify({
            "error": "No evaluation translations file found. Run 'Model Evaluation' task first."
        }), 404
        
    try:
        df = pd.read_csv(output_file, sep='\t').fillna("")
        return jsonify({
            "columns": list(df.columns),
            "data": df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load evaluation file: {str(e)}"}), 500

@app.route('/api/doc/<name>', methods=['GET'])
def get_document(name):
    """Loads markdown project documentation for web rendering."""
    # Whitelist documentation files
    doc_map = {
        "readme": "README.md",
        "architecture": "docs/ARCHITECTURE.md",
        "training": "docs/TRAINING_GUIDE.md",
        "results": "docs/RESULTS.md",
    }
    
    if name not in doc_map:
        return jsonify({"error": "Document not found"}), 404
        
    file_path = os.path.join(os.path.dirname(__file__), doc_map[name])
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"Document file {doc_map[name]} not found"}), 404
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"title": doc_map[name], "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Overwrite index.html from Flask's default template lookups
@app.route('/templates/<path:filename>')
def serve_template(filename):
    return send_from_directory('templates', filename)

if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("=" * 60)
    print("STARTING TRANSLATOR WEB INTERFACE")
    print("Visit http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5000, debug=True)
