"""Export only plots included on the currently generated homepage."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib
import json
import os
import zipfile
from PIL import Image


class PagePlots(HTMLParser):
    def __init__(self):
        super().__init__()
        self.plots=[]

    def handle_starttag(self, tag, attributes):
        attrs=dict(attributes)
        if tag in {'img','iframe'} and attrs.get('src'):
            self.plots.append((tag,attrs['src']))


def export(root=Path('.')):
    root=Path(root)
    page=PagePlots();page.feed((root/'index.html').read_text())
    assets=[];seen=set()
    for kind, url in page.plots:
        parts=urlsplit(url)
        if parts.scheme or parts.netloc:
            raise ValueError(f'External plot cannot be exported: {url}')
        path=Path(unquote(parts.path).lstrip('/'))
        if kind=='iframe':
            path=path.with_suffix('.png')
        if path.suffix.lower()!='.png' or not (root/path).is_file():
            raise ValueError(f'Missing PNG for visible plot: {url}')
        if root.resolve() not in (root/path).resolve().parents:
            raise ValueError('Plot source is outside the website')
        if path in seen:
            continue
        seen.add(path)
        assets.append((path,f'{len(assets)+1:02d}-{path.name}',kind))
    if not assets:
        raise ValueError('No visible plots found on homepage')
    destination=root/'exports';destination.mkdir(exist_ok=True)
    archive=destination/'visible-plots-png.zip';temporary=archive.with_suffix('.tmp.zip')
    entries=[]
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative,name,kind in assets:
            path=root/relative
            with Image.open(path) as image:
                if image.format!='PNG':raise ValueError(f'Not a PNG: {path}')
                size=image.size;image.verify()
            bundle.write(path,name)
            entries.append(dict(source=relative.as_posix(),export_name=name,size=list(size),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),interactive=(kind=='iframe')))
    with zipfile.ZipFile(temporary) as bundle:
        assert len(bundle.namelist())==len(assets) and bundle.testzip() is None
    temporary.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    versioned=destination/f'visible-plots-png-{len(assets)}-{digest[:10]}.zip'
    if not versioned.exists():os.link(archive,versioned)
    manifest=dict(count=len(assets),archive=versioned.name,canonical_archive=archive.name,sha256=digest,scope='Only plots displayed on the current homepage; interactive plots use their matching PNG.',images=entries)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Exported {len(assets)} currently displayed plots: {versioned}',flush=True)
    return versioned


if __name__=='__main__':
    export()
