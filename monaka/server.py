import os
import gc
import glob
import typer
import torch
from flask import Flask, request
from monaka.predictor import EnsemblePredictor, LemmaPredictor, RESC_DIR, Encoder, Decoder

app = Flask(__name__)
app.config['DEVICE'] = 'cpu'
app.config['MODEL_DIR'] = RESC_DIR
app.config['DIC_DIR'] = RESC_DIR

cmd = typer.Typer()
MODELS = dict()

@app.route("/")
def index():
    return "hello"

@app.route("/model/<modelname>/dic/<dicname>/parse", methods=['POST'])
def parse2json(modelname, dicname):
    if modelname in MODELS:
        model = MODELS[modelname]
    else:
        model_dir = os.path.join(app.config['MODEL_DIR'], modelname)
        if os.path.exists(os.path.join(model_dir, "config.json")):
            model_dir = [model_dir]
        else:
            model_dir = glob.glob(os.path.join(model_dir, "*"))
        model = EnsemblePredictor(model_dirs=model_dir, device=app.config['DEVICE'])
        MODELS[modelname] = model

    sentence = request.json.get("sentence", [''])
    if isinstance(sentence, str):
        sentence = [sentence]

    dicname = dicname.replace("--", "/")
    out = model.predict(sentence, suw_tokenizer='mecab', suw_tokenizer_option={"dic": dicname, "dic_path": app.config['DIC_DIR']}, device=app.config['DEVICE'], batch_size=1, encoder_name=request.json.get("output_format", 'jsonl'),
        node_format=request.json.get("node_format", '%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n'), 
        unk_format=request.json.get("unk_format", '%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n'), 
        eos_format=request.json.get("eos_format", 'EOS\n'), 
        bos_format=request.json.get("bos_format", '')
    )
    gc.collect()
    torch.cuda.empty_cache()
    return out
    
@cmd.command()
def run(device: str=-1, model_dir: str=RESC_DIR, dic_dir: str=RESC_DIR):
    app.config['DEVICE'] = device

    app.config['MODEL_DIR'] = model_dir
    app.config['DIC_DIR'] = dic_dir

    app.run()

if __name__ == '__main__':
    app()