"""One bounded XP004 A10 job, sharing the existing $20 budget ledger."""
import argparse,json,time,hashlib,zipfile,re
from pathlib import Path
import modal
from modal_runner import ROOT,reserve,finish,download_archive,UPPER_HOURLY_RATES,FUNCTION_TIMEOUT,STARTUP_TIMEOUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+',a.run_id):raise ValueError('Invalid run ID')
    config=json.loads(Path(a.config).read_text());config['device']='cuda'
    if not 1<=config.get('seconds',450)<=600:raise ValueError('Training time must be bounded to <=600 seconds')
    reserve(ROOT/'budget-ledger.json',a.run_id,'A10')
    app=modal.App('arc-differentiable-selfplay')
    image=(modal.Image.debian_slim(python_version='3.12')
           .pip_install('torch==2.8.0','numpy==2.5.3','pytest==8.4.2',extra_options='--timeout 120 --retries 2')
           .add_local_dir(ROOT/'arcplay','/root/project/arcplay',copy=True,ignore=['**/__pycache__/**'])
           .add_local_dir(ROOT/'tests','/root/project/tests',copy=True,ignore=['**/__pycache__/**']))
    volume=modal.Volume.from_name('arc-differentiable-selfplay-runs',create_if_missing=True)
    @app.function(image=image,gpu='A10',cpu=(2,2),memory=(8192,8192),serialized=True,
                  volumes={'/runs':volume},max_containers=1,min_containers=0,scaledown_window=2,
                  timeout=FUNCTION_TIMEOUT,startup_timeout=STARTUP_TIMEOUT,retries=0,single_use_containers=True)
    def worker(run_id,config):
        import sys,time,json,hashlib,zipfile
        from pathlib import Path
        sys.path.insert(0,'/root/project')
        from arcplay.catalogue_experiments import train
        output=Path('/runs')/run_id
        if output.exists():raise ValueError('Existing cloud run')
        started=time.perf_counter()
        if config.get('task')=='direct':
            from arcplay.grid_solver import train as train_direct
            result=train_direct(output,config['steps'],config['seconds'],'cuda')
        else:result=train(config,output)
        with zipfile.ZipFile(output/'artifacts.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                if path.name!='artifacts.zip':archive.write(path,path.name)
        with (output/'artifacts.zip').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        volume.commit()
        return dict(worker_seconds=time.perf_counter()-started,artifact_sha256=digest,steps=result['steps'],
                    scores=({k:v['overall'] for k,v in result['results'].items()} if 'results' in result else {k:result[k]['exact'] for k in ('train','heldout')}))
    try:
        with modal.enable_output(),app.run():result=worker.remote(a.run_id,config)
        finish(ROOT/'budget-ledger.json',a.run_id,status='completed',worker_seconds=result['worker_seconds'],
               estimated_usd=result['worker_seconds']/3600*UPPER_HOURLY_RATES['A10'])
        download_archive(volume,a.run_id,ROOT/'runs'/a.run_id,result['artifact_sha256'])
        print(json.dumps(result,indent=2))
    except BaseException as error:
        finish(ROOT/'budget-ledger.json',a.run_id,status='failed',error=repr(error));raise
if __name__=='__main__':main()
