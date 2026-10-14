"""Bounded XP005 neural-weight experiments using the original $20 ledger."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import modal
from modal_runner import (ROOT, reserve, finish, download_archive,
                          UPPER_HOURLY_RATES, FUNCTION_TIMEOUT, STARTUP_TIMEOUT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True);p.add_argument('--config',required=True)
    p.add_argument('--gpu',choices=('A10','L40S'),default='A10')
    a=p.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+',a.run_id):raise ValueError('Invalid run ID')
    config=json.loads(Path(a.config).read_text())
    if not 1<=config.get('seconds',450)<=600:raise ValueError('Training must be capped <=600 seconds')
    data=Path(config['data']).resolve()
    if not data.is_file():raise ValueError('Missing dataset bank')
    with data.open('rb') as stream:config['data_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    config['data']='/root/project/data.pt';config['device']='cuda'
    app=modal.App('arc-differentiable-selfplay')
    image=(modal.Image.debian_slim(python_version='3.12')
           .pip_install('torch==2.8.0','numpy==2.5.3','pytest==8.4.2',extra_options='--timeout 120 --retries 2')
           .add_local_dir(ROOT/'arcplay','/root/project/arcplay',copy=True,ignore=['**/__pycache__/**'])
           .add_local_file(data,'/root/project/data.pt',copy=True))
    resume=config.get('resume')
    if resume:
        resume=Path(resume).resolve()
        image=image.add_local_dir(resume,'/root/project/resume',copy=True,ignore=['**/source_*','**/artifacts.zip','**/examples_*'])
        config['resume']='/root/project/resume'
    volume=modal.Volume.from_name('arc-differentiable-selfplay-runs',create_if_missing=True)
    @app.function(image=image,gpu=a.gpu,cpu=(2,2),memory=(8192,8192),serialized=True,
                  volumes={'/runs':volume},max_containers=1,min_containers=0,scaledown_window=2,
                  timeout=FUNCTION_TIMEOUT,startup_timeout=STARTUP_TIMEOUT,retries=0,single_use_containers=True)
    def worker(run_id,config):
        import sys,time,json,hashlib,zipfile
        from pathlib import Path
        sys.path.insert(0,'/root/project')
        from arcplay.neural_experiments import run_fits
        output=Path('/runs')/run_id
        if output.exists():raise ValueError('Existing cloud run')
        started=time.perf_counter()
        if config.get('mode')=='latent':
            from arcplay.neural_latent import run
            result=run(config,output)
        elif config.get('mode')=='icl':
            from arcplay.neural_experiments import run_icl
            result=run_icl(config,output)
        elif config.get('mode')=='adapt':
            from arcplay.neural_experiments import run_adaptation
            result=run_adaptation(config,output)
        else:result=run_fits(config,output)
        with zipfile.ZipFile(output/'artifacts.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                if path.is_file() and path.name!='artifacts.zip':archive.write(path,path.name)
        with (output/'artifacts.zip').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        volume.commit()
        return dict(worker_seconds=time.perf_counter()-started,artifact_sha256=digest,
                    steps=result.get('steps'),macro_exact=result.get('macro_exact'))
    reserve(ROOT/'budget-ledger.json',a.run_id,a.gpu)
    try:
        with modal.enable_output(),app.run():result=worker.remote(a.run_id,config)
        finish(ROOT/'budget-ledger.json',a.run_id,status='compute_completed',worker_seconds=result['worker_seconds'],
               estimated_usd=result['worker_seconds']/3600*UPPER_HOURLY_RATES[a.gpu],artifact_sha256=result['artifact_sha256'])
        download_archive(volume,a.run_id,ROOT/'runs'/a.run_id,result['artifact_sha256'])
        finish(ROOT/'budget-ledger.json',a.run_id,status='completed')
        print(json.dumps(result,indent=2))
    except BaseException as error:
        finish(ROOT/'budget-ledger.json',a.run_id,status='failed',error=repr(error));raise


if __name__=='__main__':main()
