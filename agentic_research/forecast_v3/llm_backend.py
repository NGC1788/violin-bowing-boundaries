"""Portable JSON-only proposers: Codex, Claude Code, or file-based other AIs.

LLM output is data, never executable code. The controller validates candidates
and identity separately. CLI authentication stays in each user's own login.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


class ManualInputRequired(RuntimeError):
    pass


def validate(value, schema):
    """Strict validator for the deliberately small JSON Schema subset we emit."""
    supported = {'type','properties','required','additionalProperties','enum','items','minLength','maxLength'}
    if set(schema) - supported:
        raise ValueError('Unsupported schema keywords: ' + str(set(schema)-supported))
    types = {'object':dict,'array':list,'string':str,'boolean':bool,'integer':int,'number':(int,float)}
    kind = schema.get('type')
    if kind not in types or not isinstance(value,types[kind]) or (kind in ('integer','number') and isinstance(value,bool)):
        raise ValueError(f'Expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('Value outside enum')
    if kind == 'object':
        props = schema.get('properties',{})
        if not set(schema.get('required',[])).issubset(value):
            raise ValueError('Missing required response fields')
        if schema.get('additionalProperties') is False and set(value)-set(props):
            raise ValueError('Unexpected response fields')
        for key, child in props.items():
            if key in value:
                validate(value[key],child)
    elif kind == 'array':
        for child in value:
            validate(child,schema['items'])
    elif kind == 'string':
        if not schema.get('minLength',0) <= len(value) <= schema.get('maxLength',10**9):
            raise ValueError('Response string length outside schema')
    return value


def command_for(backend, schema, output, schema_path):
    executable = shutil.which(backend)
    if not executable:
        raise RuntimeError(f'{backend} CLI unavailable; use manual JSON mode or install/login')
    if backend == 'codex':
        return [executable,'exec','--sandbox','read-only','--ephemeral','--ignore-user-config',
                '--skip-git-repo-check','--color','never','--json','--output-schema',str(schema_path),
                '--output-last-message',str(output),'-C',str(output.parent),'-']
    if backend == 'claude':
        return [executable,'--print','--output-format','json','--json-schema',json.dumps(schema),
                '--tools','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
                '--no-session-persistence','--setting-sources','','--permission-mode','dontAsk',
                '--disable-slash-commands']
    raise ValueError('Unknown backend')


def parse_claude(payload):
    if payload.get('is_error') or payload.get('subtype') not in (None,'success'):
        raise RuntimeError('Claude returned an error; inspect the saved event log')
    if isinstance(payload.get('structured_output'),dict):
        return payload['structured_output']
    # Older CLI releases may return the exact JSON text in result.
    if isinstance(payload.get('result'),str):
        return json.loads(payload['result'])
    raise ValueError('No structured Claude response')


def run_structured(prompt, schema, directory, role, backend='codex', timeout=180, manual_dir=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    schema_path = directory / f'{role}_schema.json'
    output = directory / f'{role}_response.json'
    schema_path.write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
    (directory/f'{role}_prompt.txt').write_text(prompt)
    request = {'backend':backend,'role':role,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
               'schema_sha256':hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()}
    (directory/f'{role}_request.json').write_text(json.dumps(request,indent=2)+'\n')
    started = time.monotonic()
    if backend == 'manual':
        source = Path(manual_dir)/f'{role}_response.json' if manual_dir else output
        if not source.is_file():
            raise ManualInputRequired(f'Read {directory/str(role+"_prompt.txt")} and supply JSON at {source}')
        value = validate(json.loads(source.read_text()),schema)
    else:
        command = command_for(backend,schema,output,schema_path)
        # No shell interpolation. Logs never include auth-status/keyring content.
        proc = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,text=True,cwd=directory,start_new_session=True)
        try:
            stdout,stderr = proc.communicate(prompt,timeout=timeout)
        except subprocess.TimeoutExpired:
            import os,signal
            os.killpg(proc.pid,signal.SIGKILL)
            stdout,stderr = proc.communicate()
            (directory/f'{role}_events.log').write_text(stdout+'\nSTDERR\n'+stderr)
            raise
        (directory/f'{role}_events.log').write_text(stdout+'\nSTDERR\n'+stderr)
        if proc.returncode:
            raise RuntimeError(f'{backend} returned {proc.returncode}; inspect saved logs')
        value = json.loads(output.read_text()) if backend=='codex' else parse_claude(json.loads(stdout))
        validate(value,schema)
    output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (directory/f'{role}_call.json').write_text(json.dumps({**request,'elapsed_seconds':time.monotonic()-started,
        'response_sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2)+'\n')
    return value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['codex','claude','manual'],required=True)
    ap.add_argument('--prompt',type=Path,required=True)
    ap.add_argument('--schema',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--role',choices=['proposer','reviewer'],default='proposer')
    ap.add_argument('--manual-dir',type=Path)
    ap.add_argument('--timeout',type=int,default=180)
    a=ap.parse_args()
    value=run_structured(a.prompt.read_text(),json.loads(a.schema.read_text()),a.output_dir,a.role,a.backend,a.timeout,a.manual_dir)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    main()
