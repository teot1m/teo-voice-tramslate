import asyncio,json,time,resource,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from uvt.engines.tts_moss_onnx import MossOnnxTTS

async def main():
 out=Path('artifacts/verification/moss-voices');out.mkdir(parents=True,exist_ok=True)
 model_dir=Path('.models/moss-tts');manifest=json.loads((model_dir/'browser_poc_manifest.json').read_text())
 voices={v['voice']:v for v in manifest['builtin_voices']}
 cfg=SimpleNamespace(model_path=str(model_dir),codec_path='.models/moss-codec',voice_id='Adam',threads=4)
 engine=MossOnnxTTS(cfg)
 start=time.perf_counter();await engine.warmup();load=time.perf_counter()-start
 text='Здравствуйте! Выберите удобный голос для перевода.'
 result={'engine':'moss-onnx','target_language':'ru','text':text,'runtime':'ONNX CPU, 4 threads','warmup_seconds':round(load,3),'tts_source':'OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX','tts_revision':'f52645cb467506d8e18e746ddd59482685b74e58','codec_source':'OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX','codec_revision':'ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae','manifest_sha256':hashlib.sha256((model_dir/'browser_poc_manifest.json').read_bytes()).hexdigest(),'network':'HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1','voices':[]}
 try:
  for name in ['Adam','Nathan','Ava','Bella']:
   cfg.voice_id=name
   start=time.perf_counter();audio,rate=await engine.synthesize(text,'ru');elapsed=time.perf_counter()-start
   assert np.isfinite(audio).all() and len(audio)>0 and float(np.max(np.abs(audio)))>0.001
   file=out/(name.lower()+'-ru.wav');sf.write(file,audio,rate,subtype='PCM_16')
   item={'id':name,'official_label':voices[name]['display_name'],'official_group':voices[name]['group'],'reference_file':voices[name]['audio_file'],'output_file':file.name,'synthesis_seconds':round(elapsed,3),'audio_seconds':round(len(audio)/rate,3),'real_time_factor':round(elapsed/(len(audio)/rate),3),'sample_rate':rate,'peak':round(float(np.max(np.abs(audio))),4)}
   result['voices'].append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
  result['peak_rss_mib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,1)
 finally:
  await engine.close()
 result['model_closed']=True
 (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('DONE',json.dumps({'warmup_seconds':result['warmup_seconds'],'peak_rss_mib':result['peak_rss_mib'],'voices':len(result['voices']),'model_closed':True}),flush=True)

asyncio.run(main())
