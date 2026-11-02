# Fase 5E — localizar a inicializacao ORIGINAL do CFIS no Drive.
# Somente leitura de codigo .py/.ipynb/.txt; nao acessa datasets nem testes.
from pathlib import Path
import json

ROOT = Path('/content/drive/MyDrive/CFIS_Artigo_Diogo')
NEEDLES = ('CFIS_feature_selection', 'consolidated_permutation_adapter',
           'equivalent_1e12', 'exact_cfis', 'cfis_function', 'original_permutation')
EXTENSIONS = {'.py', '.ipynb', '.txt'}
MAX_FILE_MB = 15
found = []
print('FASE 5E — LOCALIZANDO A INICIALIZACAO ORIGINAL DO CFIS (SEM EXECUTAR CODIGO)')
if not ROOT.is_dir():
    raise FileNotFoundError(f'Pasta do projeto nao encontrada: {ROOT}')
for path in ROOT.rglob('*'):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue
    try:
        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            continue
        if path.suffix.lower() == '.ipynb':
            nb = json.loads(path.read_text(encoding='utf-8'))
            cells = nb.get('cells', [])
            matches = []
            for i, cell in enumerate(cells, 1):
                if cell.get('cell_type') != 'code':
                    continue
                source = ''.join(cell.get('source', []))
                hits = [x for x in NEEDLES if x in source]
                if hits:
                    matches.append((i, hits))
            if matches:
                found.append((str(path), matches))
        else:
            source = path.read_text(encoding='utf-8', errors='replace')
            hits = [x for x in NEEDLES if x in source]
            if hits:
                found.append((str(path), [('arquivo', hits)]))
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        print(f'AVISO: nao foi possivel ler {path.name}: {type(exc).__name__}')
for path, matches in found:
    print('\nARQUIVO:', path)
    for where, hits in matches:
        print('  Celula/posicao:', where, '| referencias:', ', '.join(hits))
print('\nTotal de arquivos candidatos:', len(found))
print('Nenhum codigo localizado foi executado; nenhum dataset/teste foi acessado.')
print('Proximo: executar as celulas ORIGINAIS de implementacao e equivalencia; nao forcar flags.')
