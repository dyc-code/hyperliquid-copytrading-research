"""Verify that the cached audit, raw input and frozen experiment share a source."""
from pathlib import Path
import hashlib
import json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_source_binding(source, audit_file, summary_file, config_file, protocol_file):
    provenance_file = Path(audit_file).with_name('audit_cache_provenance.json')
    if not provenance_file.is_file():
        raise RuntimeError('Audit cache provenance missing; rebuild the audit before selection')
    provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
    summary = json.loads(Path(summary_file).read_text(encoding='utf-8'))
    actual = {'source_sha256': digest(source), 'cache_sha256': digest(audit_file),
              'summary_sha256': digest(summary_file)}
    if (provenance.get('version') != 1
            or provenance.get('cache_file') != Path(audit_file).name
            or provenance.get('summary_file') != Path(summary_file).name
            or any(provenance.get(k) != value for k, value in actual.items())
            or summary['summary'].get('sha256') != actual['source_sha256']):
        raise RuntimeError('Raw source and audit cache provenance do not match; rebuild in a new experiment')
    if Path(protocol_file).exists():
        frozen = json.loads(Path(protocol_file).read_text(encoding='utf-8'))
        if (frozen.get('source_sha256') != actual['source_sha256']
                or frozen.get('configuration_sha256') != digest(config_file)):
            raise RuntimeError('Frozen source/configuration changed; create a separately named experiment')
    return actual
