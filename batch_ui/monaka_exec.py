import threading
import uuid
import os
import json
import subprocess
import glob
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.join(BASE_DIR, "config")
WORK_DIR = os.path.join(BASE_DIR, "work")
RESL_DIR = os.path.join(BASE_DIR, "results")
ZIPF_DIR = os.path.join(BASE_DIR, "zipfiles")

class MonakaExec:

    def __init__(self):
        self.threads = dict()

    def load_conf(self, conf_path: str) -> dict:
        with open(os.path.join(CONF_DIR, conf_path)) as f:
            conf = json.load(f)
        return conf
    
    def create_command(self, conf: dict, input_:str, output_: str, stdout: str) -> str:
        command = conf["exec"]
        kargs = [ f"{k} {v}" for k, v in conf.items() if k.startswith("-")]
        args = [ (k,v) for k, v in conf.items() if k.startswith("args")]
        args.sort(key=lambda v: v[0])
        command = f"{command} {' '.join(kargs)}"
        args_ = [v for _, v in args]
        command = f"{command} {' '.join(args_)}"

        if '{output}' not in command:
            return command.format(input=input_, output=output_), output_
        else:
            return command.format(input=input_, output=output_), stdout


    def run(self, conf_path: str, input_: str):
        id_ = str(uuid.uuid1())
        conf = self.load_conf(conf_path)

        out_dir = os.path.join(RESL_DIR, id_)
        os.mkdir(out_dir)
        out_path = os.path.join(out_dir, "output")
        stdout_path = os.path.join(out_dir, "stdout")
        run_js = os.path.join(out_dir, "run.json")
        with open(run_js, 'w') as f:
            json.dump({"config": conf_path, "input": os.path.basename(input_), "start": datetime.datetime.now().isoformat()}, f)

        command, out = self.create_command(conf, input_, out_path, stdout_path)


        def _inner_run():
            with open(out, 'w') as f:
                subprocess.run(command.split(), stdout=f)

        th = threading.Thread(target=_inner_run, daemon=True)
        self.threads[id_] = th
        th.start()

        return id_
    
    def status(self) -> dict:
        res = dict()
        for name in glob.glob(os.path.join(RESL_DIR, "*")):
            id_ = os.path.basename(name)
            
            run_js = os.path.join(name, "run.json")
            with open(run_js) as f:
                d = json.load(f)
            
            res[id_] = d
            if id_ not in self.threads:
                res[id_]["status"] = "closed"
            else:
                if self.threads[id_].is_alive():
                    res[id_]["status"] = "running"
                else:
                    res[id_]["status"] = "finished"
        return res
