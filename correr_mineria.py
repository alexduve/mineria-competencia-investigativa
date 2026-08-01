#!/usr/bin/env python3
"""Construye y EJECUTA un notebook de minería sobre la carpeta de revisión, usando el kernel real.

Uso:
    python3 correr_mineria.py "<revision_dir>"     # la carpeta con manifest.json (la crea extraer_ensayos)

Por qué notebook y no script: la regla de Alex es que lo cuantitativo corra en Jupyter con librerías
reales y quede AUDITABLE. Este script genera <revision_dir>/analisis_mineria.ipynb, lo ejecuta con
nbconvert (kernel python3 = /usr/bin/python3, el del stack científico) y deja el notebook ejecutado
con la tabla y las figuras embebidas. Los artefactos (metricas/*.json, codificacion.json,
cohorte.json, tabla_metricas.json, figuras/*.png) los consume después el reporte.

Si algo del entorno falta (modelo es de spaCy, etc.), el notebook lo intenta resolver y, si no,
mineria_core degrada con método declarado. Falla con exit!=0 solo si el kernel revienta de verdad.
"""
import os, sys, json, subprocess

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def construir_notebook(revision_dir):
    setup = f"""
import sys, os, json, subprocess
sys.path.insert(0, {SCRIPTS!r})
REVISION_DIR = {revision_dir!r}

# Mejor calidad de spaCy si el modelo es-md está disponible; si no, mineria_core degrada.
try:
    import spacy
    try:
        spacy.load("es_core_news_md")
    except Exception:
        try:
            # Entorno PEP 668 (externally-managed): pip directo lo bloquea, por eso break-system-packages.
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "--break-system-packages",
                            "https://github.com/explosion/spacy-models/releases/download/"
                            "es_core_news_md-3.8.0/es_core_news_md-3.8.0-py3-none-any.whl"],
                           check=False, timeout=600)
        except Exception as e:
            print("spaCy es-md no disponible, se usará fallback:", e)
except Exception as e:
    print("spaCy ausente, fallback regex:", e)
print("Entorno listo.")
""".strip()

    correr = """
import mineria_core, importlib
importlib.reload(mineria_core)
resumen = mineria_core.run(REVISION_DIR)
print(json.dumps(resumen["componentes"], ensure_ascii=False))
print("documentos:", resumen["n_documentos"],
      "| codificación:", resumen["codificacion"],
      "| cohorte:", resumen["tiene_cohorte"])
""".strip()

    tabla = """
import pandas as pd
df = pd.DataFrame(resumen["tabla"]).set_index("documento")
pd.set_option("display.max_columns", None, "display.width", 200)
df
""".strip()

    figuras = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs(os.path.join(REVISION_DIR, "figuras"), exist_ok=True)
metricas_plot = ["ttr", "mtld", "fernandez_huerta", "densidad_lexica",
                 "conectores_x100", "coherencia_semantica", "citas_densidad_1000"]
disp = [m for m in metricas_plot if m in df.columns and df[m].notna().any()]
n = len(disp)
if n:
    fig, axes = plt.subplots(1, n, figsize=(3.1*n, 3.2))
    if n == 1: axes = [axes]
    for ax, m in zip(axes, disp):
        vals = df[m].dropna()
        if len(vals) >= 2:
            ax.hist(vals, bins=min(10, len(vals)), color="#2C6E7F", edgecolor="white")
            for x in vals: ax.axvline(x, color="#D98C00", alpha=.35, lw=1)
        else:
            ax.bar([df.index[0][:14]], [vals.iloc[0]], color="#2C6E7F")
        ax.set_title(m, fontsize=10); ax.tick_params(labelsize=8)
    plt.tight_layout()
    out = os.path.join(REVISION_DIR, "figuras", "panorama_metricas.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); print("figura:", out)
    plt.show()
else:
    print("Sin métricas suficientes para figura.")
""".strip()

    codigos = """
cod = json.load(open(os.path.join(REVISION_DIR, "codificacion.json"), encoding="utf-8"))
print("Método de codificación:", cod.get("metodo"))
for c in cod.get("codigos", []):
    print(f"\\n[código {c['codigo_id']}] {' · '.join(c['etiqueta_terminos'])}  "
          f"({c['n_segmentos']} segmentos, {c['n_documentos']} docs)")
    print("  ejemplo:", c["cita_representativa"][:200])
""".strip()

    redes = """
from IPython.display import Image, display
redes = json.load(open(os.path.join(REVISION_DIR, "redes.json"), encoding="utf-8"))
for stem, r in redes.items():
    print(f"\\n=== Red de conceptos · {stem} ===")
    print("nodos:", r.get("nodos"), "| aristas:", r.get("aristas"),
          "| umbral:", r.get("umbral_arista"))
    print("conceptos centrales:", [c["concepto"] for c in r.get("conceptos_centrales", [])])
    print("entidades:", r.get("entidades"))
    print("¿autores/entidades que dialogan (co-ocurren)?:", r.get("aristas_entre_entidades"))
    png = os.path.join(REVISION_DIR, "figuras", r.get("png", ""))
    if r.get("png") and os.path.exists(png):
        display(Image(filename=png))
""".strip()

    nb = new_notebook(cells=[
        new_markdown_cell("# Minería cuantitativa de tesis / trabajos finales\n"
                          "Ejecutado con el kernel real. Artefactos → `metricas/`, `codificacion.json`, "
                          "`cohorte.json`, `tabla_metricas.json`, `figuras/`."),
        new_code_cell(setup),
        new_code_cell(correr),
        new_markdown_cell("## Tabla de métricas por documento"),
        new_code_cell(tabla),
        new_markdown_cell("## Panorama de métricas"),
        new_code_cell(figuras),
        new_markdown_cell("## Códigos emergentes (grounded)"),
        new_code_cell(codigos),
        new_markdown_cell("## Red de conceptos (co-ocurrencia)"),
        new_code_cell(redes),
    ])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: correr_mineria.py <revision_dir>")
    revision_dir = os.path.abspath(sys.argv[1])
    if not os.path.isfile(os.path.join(revision_dir, "manifest.json")):
        sys.exit(f"No hay manifest.json en {revision_dir} (corre antes extraer_ensayos.py)")

    nb = construir_notebook(revision_dir)
    nb_path = os.path.join(revision_dir, "analisis_mineria.ipynb")
    nbformat.write(nb, nb_path)
    print(f"Notebook → {nb_path}\nEjecutando con el kernel python3 (puede tardar la 1ª vez por el "
          f"modelo de embeddings)…")

    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": revision_dir}})
    except CellExecutionError as e:
        nbformat.write(nb, nb_path)
        sys.exit(f"[FALLO] El notebook reventó: {e}\nRevisa {nb_path} para ver el traceback.")
    nbformat.write(nb, nb_path)

    print("\nOK · minería ejecutada")
    for art in ["tabla_metricas.json", "codificacion.json", "cohorte.json"]:
        p = os.path.join(revision_dir, art)
        print(f"   {'✓' if os.path.exists(p) else '–'} {art}")
    print(f"   notebook ejecutado → {nb_path}")
    print(f"   métricas por doc   → {os.path.join(revision_dir, 'metricas')}/")


if __name__ == "__main__":
    main()
