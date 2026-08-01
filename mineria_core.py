#!/usr/bin/env python3
"""Motor de minería cuantitativa para tesis / trabajos finales.

ESTE MÓDULO SE EJECUTA DENTRO DE JUPYTER (vía correr_mineria.py → nbconvert). No lo corras a mano
para "leer números": su salida son artefactos JSON + figuras que después Claude interpreta y ancla a
RAG. La regla de Alex es que lo cuantitativo lo haga el kernel con librerías reales y reporte
incertidumbre, no Claude a ojo.

Qué calcula (por documento):
  - Conteos, diversidad léxica (TTR, Guiraud, MTLD), densidad léxica (POS si hay spaCy).
  - Legibilidad en español (Fernández-Huerta e INFLESZ/Szigriszt-Pazos) + nivel.
  - Cohesión: densidad de conectores + cohesión léxica entre párrafos contiguos + (si hay
    sentence-transformers) coherencia semántica entre oraciones contiguas.
  - Aparato crítico: extracción y parseo de citas (Autor, año, p. X), densidad, citas malformadas
    (p. ej. falta el año), cruce cita↔lista de referencias (huérfanas / no citadas), citas de
    segunda mano.
Codificación emergente (grounded, data-driven) sobre los párrafos:
  - Embeddings (sentence-transformers) o TF-IDF → clustering (HDBSCAN, fallback KMeans) → códigos
    emergentes con términos TF-IDF representativos, cita ejemplo, frecuencia y co-ocurrencia.
Si el lote trae ≥2 documentos: percentiles y z-scores de cada uno contra el grupo, con IC bootstrap
de las medias del grupo (scipy). Con n<8 se marca `muestra_pequena` para que el reporte sea honesto
sobre la incertidumbre.

Dependencias: todas presentes en el kernel salvo, a veces, el modelo `es` de spaCy y textstat; el
notebook las resuelve. Aquí todo es degradable: si falta algo, baja a un método más simple y lo
declara en el JSON (`metodo`), nunca falla en silencio.
"""
import os, re, json, math, glob, unicodedata
from collections import Counter, defaultdict

import numpy as np

# ----- Optativos (degradación elegante) -----
try:
    import spacy
    try:
        _NLP = spacy.load("es_core_news_md", disable=["ner"])
    except Exception:
        try:
            _NLP = spacy.load("es_core_news_sm", disable=["ner"])
        except Exception:
            _NLP = None
except Exception:
    _NLP = None

try:
    from sentence_transformers import SentenceTransformer
    _EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
except Exception:
    _EMB = None

# ---------------------------------------------------------------------------
STOP_ES = set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
dos el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba estamos
estan estar este esto estos fin fue fueron ha haber habia han hasta hay la las le les lo los mas me mi
mis mucho muchos muy nada ni no nos nosotros o os otra otro para pero poco por porque que quien se sea
sean segun ser si sin sobre solo son su sus tal tambien tan tanto te tiene tienen toda todas todo todos
tras tu tus un una unas uno unos vosotros y ya yo era ese esa este esta del al lo la las los un una""".split())

CONECTORES = [
    "por lo tanto", "sin embargo", "no obstante", "en consecuencia", "por consiguiente",
    "en cambio", "mientras que", "dado que", "puesto que", "ya que", "debido a", "por ello",
    "en efecto", "es decir", "en otras palabras", "por ejemplo", "en resumen", "en conclusion",
    "en conclusión", "finalmente", "en primer lugar", "por otra parte", "de hecho", "asi pues",
    "así pues", "ademas", "además", "asimismo", "a pesar de", "en contraste", "de igual manera",
    "a partir de", "en virtud de", "de modo que", "de manera que", "por su parte", "aunque",
    "por consiguiente", "en este sentido", "cabe destacar", "por un lado", "por el contrario",
]

VOCALES = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")


def _strip(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


def _palabras(texto):
    return re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", texto)


def _silabas(palabra):
    """Conteo aproximado de sílabas en español: grupos de vocales (diptongos = 1)."""
    grupos = re.findall(r"[aeiouáéíóúüAEIOUÁÉÍÓÚÜ]+", palabra)
    return max(1, len(grupos)) if palabra else 0


def _oraciones(texto):
    if _NLP is not None:
        doc = _NLP(texto[:1_000_000])
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    # Fallback regex
    partes = re.split(r"(?<=[.!?¿¡])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", texto)
    return [p.strip() for p in partes if len(p.strip()) > 1]


RE_REFS_HEAD = re.compile(
    r"(?:^|\n)\s*#{0,3}\s*(referencias|bibliograf[íi]a|obras citadas|fuentes consultadas)\b"
    r"[^\n]{0,30}(?:\n|$)", re.IGNORECASE)  # admite "Referencias bibliográficas:", etc.


def separar_referencias(texto):
    """Devuelve (cuerpo, bloque_referencias). El cuerpo es lo que se mina/codifica; el bloque de
    referencias solo se usa para el cruce cita↔referencias (no debe contaminar clusters ni citas)."""
    m = RE_REFS_HEAD.search(texto)
    if m:
        return texto[:m.start()], texto[m.end():]
    return texto, ""


def cuerpo_analizable(texto):
    """Texto para minar: sin portada institucional ni lista de referencias. La portada (carátula con
    líneas en MAYÚSCULAS, nombre de la escuela, autor, materia) distorsiona léxico/legibilidad y hace
    que palabras como SUJETO/FORMACIÓN parezcan entidades. Heurística general: el cuerpo real empieza
    en el primer párrafo de prosa (>=25 palabras)."""
    cuerpo, _refs = separar_referencias(texto)
    paras = cuerpo.split("\n\n")
    inicio = 0
    for i, p in enumerate(paras):
        if len(p.split()) >= 25:
            inicio = i
            break
    return "\n\n".join(paras[inicio:]).strip() or cuerpo


def _parrafos(texto):
    out = []
    for bloque in texto.split("\n\n"):
        b = bloque.strip()
        if not b or b.startswith("## "):
            continue
        if len(b.split()) >= 5:   # ignora líneas sueltas/encabezados
            out.append(b)
    return out


# ---------- Diversidad léxica ----------
def mtld(tokens, umbral=0.72):
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010), promedio bidireccional."""
    def _pasada(seq):
        factores, tipos, n = 0, set(), 0
        for t in seq:
            n += 1
            tipos.add(t)
            ttr = len(tipos) / n
            if ttr <= umbral:
                factores += 1
                tipos, n = set(), 0
        if n > 0:
            ttr = len(tipos) / n
            factores += (1 - ttr) / (1 - umbral)
        return len(seq) / factores if factores else float(len(seq))
    if len(tokens) < 50:
        return None  # MTLD no es confiable en textos cortos
    return round((_pasada(tokens) + _pasada(tokens[::-1])) / 2, 1)


def mattr(tokens, w=50):
    """Moving-Average TTR (Covington & McFall): TTR en ventana deslizante fija → diversidad léxica
    INVARIANTE a la longitud (a diferencia de TTR/Guiraud). Ideal para comparar trabajos de distinto
    tamaño en un lote."""
    if not tokens:
        return None
    if len(tokens) <= w:
        return round(len(set(tokens)) / len(tokens), 4)
    ttrs = [len(set(tokens[i:i + w])) / w for i in range(len(tokens) - w + 1)]
    return round(sum(ttrs) / len(ttrs), 4)


def metricas_lexicas(texto):
    pals = [p.lower() for p in _palabras(texto)]
    n = len(pals)
    tipos = len(set(pals))
    out = {
        "n_tokens": n,
        "n_tipos": tipos,
        "ttr": round(tipos / n, 4) if n else None,
        "guiraud": round(tipos / math.sqrt(n), 2) if n else None,  # Root TTR (estable vs longitud)
        "mtld": mtld(pals),
        "mattr": mattr(pals),   # diversidad léxica invariante a la longitud
    }
    # Densidad léxica: contenido / total. Con spaCy = POS; sin spaCy = no-stopword.
    if _NLP is not None and n:
        doc = _NLP(texto[:1_000_000])
        cont = sum(1 for t in doc if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"))
        tot = sum(1 for t in doc if t.is_alpha)
        out["densidad_lexica"] = round(cont / tot, 4) if tot else None
        out["metodo_densidad"] = "POS (spaCy)"
    elif n:
        cont = sum(1 for p in pals if p not in STOP_ES)
        out["densidad_lexica"] = round(cont / n, 4)
        out["metodo_densidad"] = "no-stopword (sin spaCy)"
    return out


# ---------- Legibilidad ----------
def metricas_legibilidad(texto):
    oraciones = _oraciones(texto)
    pals = _palabras(texto)
    n_pal, n_ora = len(pals), max(1, len(oraciones))
    n_sil = sum(_silabas(p) for p in pals)
    spw = n_sil / n_pal if n_pal else 0      # sílabas por palabra
    wps = n_pal / n_ora                       # palabras por oración
    fh = 206.84 - 60 * spw - 1.02 * wps                  # Fernández-Huerta
    inflesz = 206.835 - 62.3 * spw - wps                 # Szigriszt-Pazos / INFLESZ
    if fh >= 80:   nivel = "muy fácil"
    elif fh >= 65: nivel = "normal"
    elif fh >= 50: nivel = "algo difícil"
    elif fh >= 30: nivel = "difícil (académico)"
    else:          nivel = "muy difícil"
    largas = sum(1 for o in oraciones if len(_palabras(o)) > 30)
    return {
        "n_oraciones": len(oraciones),
        "palabras_por_oracion": round(wps, 1),
        "silabas_por_palabra": round(spw, 2),
        "fernandez_huerta": round(fh, 1),
        "inflesz": round(inflesz, 1),
        "nivel_lectura": nivel,
        "oraciones_largas_pct": round(100 * largas / n_ora, 1),
        "metodo_oraciones": "spaCy" if _NLP is not None else "regex",
    }


# ---------- Cohesión / coherencia ----------
def _bow(tokens):
    c = Counter(t for t in tokens if t not in STOP_ES and len(t) > 2)
    return c


def _cos_bow(a, b):
    comunes = set(a) & set(b)
    if not comunes:
        return 0.0
    num = sum(a[t] * b[t] for t in comunes)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def metricas_cohesion(texto):
    oraciones = _oraciones(texto)
    n_ora = max(1, len(oraciones))
    low = _strip(texto)
    n_con = sum(low.count(_strip(c)) for c in CONECTORES)
    out = {
        "conectores_por_100_oraciones": round(100 * n_con / n_ora, 1),
        "n_conectores": n_con,
    }
    # Cohesión léxica entre párrafos contiguos (bag-of-words coseno)
    paras = _parrafos(texto)
    if len(paras) >= 2:
        bows = [_bow([w.lower() for w in _palabras(p)]) for p in paras]
        sims = [_cos_bow(bows[i], bows[i + 1]) for i in range(len(bows) - 1)]
        out["cohesion_lexica_parrafos"] = round(float(np.mean(sims)), 3)
        out["saltos_tematicos"] = int(sum(1 for s in sims if s < 0.05))
    # Coherencia semántica (local entre oraciones contiguas + global all-pairs + deriva)
    if _EMB is not None and len(oraciones) >= 3:
        try:
            E = _EMB.encode(oraciones, normalize_embeddings=True, show_progress_bar=False)
            local = [float(np.dot(E[i], E[i + 1])) for i in range(len(E) - 1)]
            S = E @ E.T
            glob = float((S.sum() - np.trace(S)) / (len(E) * (len(E) - 1)))  # media all-pairs
            centro = E.mean(0)
            centro = centro / (np.linalg.norm(centro) + 1e-9)
            given = E @ centro                                   # "givenness" por oración
            deriva = float(np.polyfit(range(len(E)), E @ E[0], 1)[0])  # pendiente vs apertura
            out["coherencia_semantica"] = round(float(np.mean(local)), 3)   # local (ya existía)
            out["coherencia_min"] = round(float(np.min(local)), 3)
            out["coherencia_global"] = round(glob, 3)
            out["ratio_global_local"] = round(glob / (np.mean(local) + 1e-9), 3)  # <1 = deriva
            out["deriva_tematica"] = round(deriva, 4)            # negativa = se aleja de la apertura
            out["givenness_media"] = round(float(np.mean(given)), 3)
            out["metodo_coherencia"] = "sentence-transformers"
        except Exception as e:
            out["metodo_coherencia"] = f"no disponible ({type(e).__name__})"
    else:
        out["metodo_coherencia"] = "no disponible (sin embeddings o texto corto)"
    return out


# ---------- Complejidad sintáctica (Pack B · spaCy) ----------
_SUB_DEPS = {"acl", "advcl", "ccomp", "xcomp", "csubj", "relcl", "acl:relcl"}
_SUF_NOM = ("ción", "sión", "miento", "dad", "encia", "anza", "aje", "ura", "ismo", "ncia")


def _prof_arbol(tok):
    d = 0
    while tok.head != tok and d < 100:
        tok = tok.head
        d += 1
    return d


def metricas_sintaxis(texto):
    """Complejidad sintáctica por análisis de dependencias (spaCy). Ortogonal a la legibilidad: mide
    estructura, no sílabas. Sin spaCy devuelve método 'no disponible'."""
    if _NLP is None:
        return {"metodo": "no disponible (sin spaCy)"}
    doc = _NLP(texto[:1_000_000])
    # MDD: distancia media de dependencia (carga de procesamiento sintáctico)
    dists = [abs(t.i - t.head.i) for t in doc if t.head != t and t.dep_ != "punct"]
    # Subordinación: cláusulas subordinadas / cláusulas (verbos finitos)
    sub = sum(1 for t in doc if t.dep_ in _SUB_DEPS)
    finitos = sum(1 for t in doc if t.pos_ == "VERB" and "Fin" in t.morph.get("VerbForm"))
    # Profundidad del árbol por oración (máximo) y palabras antes del verbo raíz (left-embeddedness)
    profs, left = [], []
    for sent in doc.sents:
        toks = [t for t in sent if t.is_alpha]
        if toks:
            profs.append(max(_prof_arbol(t) for t in sent))
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root is not None:
            left.append(sum(1 for t in sent if t.i < root.i and t.is_alpha))
    # Nominalización y modificadores por sustantivo
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    nomin = sum(1 for t in nouns if t.lemma_.lower().endswith(_SUF_NOM))
    mods = [sum(1 for c in t.children if c.dep_ in {"amod", "nmod", "det"}) for t in nouns]
    # Densidad proposicional (idea density, estilo CPIDR adaptado a POS español)
    PROP = {"VERB", "ADJ", "ADV", "ADP", "SCONJ", "CCONJ"}
    props = sum(1 for t in doc if t.pos_ in PROP) - sum(1 for t in doc if t.pos_ == "AUX")
    palabras = sum(1 for t in doc if t.is_alpha)
    return {
        "mdd": round(float(np.mean(dists)), 2) if dists else None,
        "indice_subordinacion": round(sub / finitos, 3) if finitos else None,
        "profundidad_arbol_media": round(float(np.mean(profs)), 2) if profs else None,
        "palabras_antes_verbo": round(float(np.mean(left)), 1) if left else None,
        "ratio_nominalizacion": round(nomin / len(nouns), 3) if nouns else None,
        "modificadores_por_sustantivo": round(float(np.mean(mods)), 2) if mods else None,
        "densidad_proposicional": round(props / palabras, 3) if palabras else None,
        "metodo": "spaCy dependency parse",
    }


# ---------- Argumentación / respaldo (Pack A · heurístico) ----------
_CAUSAL = ["porque", "por lo tanto", "por ello", "por consiguiente", "en consecuencia", "dado que",
           "puesto que", "ya que", "debido a", "de ahí que", "por eso", "así que", "de modo que",
           "de manera que", "de tal forma que"]
_ADVERS = ["sin embargo", "no obstante", "en cambio", "aunque", "a pesar de", "por el contrario",
           "ahora bien", "pese a", "si bien"]


def metricas_argumentacion(texto):
    """Índice de respaldo argumentativo (heurístico, no clasificación supervisada): proporción de
    oraciones que llevan respaldo (cita o conector causal) y densidad de conectores argumentativos
    (causales + adversativos), que distingue prosa razonada de aserción enumerativa."""
    oraciones = _oraciones(texto)
    n = max(1, len(oraciones))
    con_cita, con_causal, con_advers = 0, 0, 0
    for o in oraciones:
        low = _strip(o)
        tiene_cita = bool(RE_PAREN.search(o) and RE_ANIO.search(o)) or bool(RE_NARR.search(o))
        tiene_causal = any(_strip(c) in low for c in _CAUSAL)
        tiene_advers = any(_strip(c) in low for c in _ADVERS)
        con_cita += tiene_cita
        con_causal += tiene_causal
        con_advers += tiene_advers
    con_respaldo = sum(1 for o in oraciones
                       if (RE_PAREN.search(o) and RE_ANIO.search(o)) or RE_NARR.search(o)
                       or any(_strip(c) in _strip(o) for c in _CAUSAL))
    return {
        "n_oraciones": len(oraciones),
        "pct_oraciones_con_respaldo": round(100 * con_respaldo / n, 1),
        "oraciones_con_cita": con_cita,
        "densidad_causal_x100": round(100 * con_causal / n, 1),
        "densidad_adversativa_x100": round(100 * con_advers / n, 1),
        "nota": "Heurístico (cita o conector causal). Índice de respaldo, no clasificación validada.",
    }


# ---------- Aparato crítico (citas) ----------
RE_ANIO = re.compile(r"\b(1[89]\d{2}|20\d{2})\b|s\.?\s*f\.?")
RE_PAG = re.compile(r"\bpp?\.?\s*\d+|\bp[áa]g(?:ina|s)?\.?\s*\d+", re.IGNORECASE)
RE_PAREN = re.compile(r"\(([^()]{1,180})\)")
RE_NARR = re.compile(
    r"([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:y|&|et\s+al\.?,?)\s*[A-ZÁÉÍÓÚÑ]?[\wáéíóúñ]*)?)\s*"
    r"\(\s*(1[89]\d{2}|20\d{2}|s\.?\s*f\.?)[^)]*\)")
RE_SEGUNDA = re.compile(
    r"\bcit(?:ó|o|ada|ado)\s+(?:en|por)\b|\bas cited in\b|\bcomo se cit[oó] en\b",
    re.IGNORECASE)


def _autor_clave(cadena):
    cadena = cadena.strip().strip(",")
    cadena = re.split(r"[,(]", cadena)[0]
    tok = cadena.split()
    return _strip(tok[0]) if tok else None


def metricas_citas(texto):
    cuerpo = cuerpo_analizable(texto)            # citas en-texto: cuerpo sin portada ni referencias
    _, refs_bloque = separar_referencias(texto)  # lista de referencias para el cruce
    paren = []   # citas entre paréntesis (autor_clave, tiene_anio, tiene_pag, raw)
    for m in RE_PAREN.finditer(cuerpo):
        contenido = m.group(1)
        tiene_anio = bool(RE_ANIO.search(contenido))
        tiene_pag = bool(RE_PAG.search(contenido))
        # parece cita si tiene año o página y empieza con mayúscula (autor)
        if (tiene_anio or tiene_pag) and re.match(r"\s*[A-ZÁÉÍÓÚÑ]", contenido):
            paren.append((_autor_clave(contenido), tiene_anio, tiene_pag, m.group(0)))
    narr = [m.group(0) for m in RE_NARR.finditer(cuerpo)]   # narrativas (cuentan, pero ruidosas)

    n_pal = len(_palabras(cuerpo))
    n_citas = len(paren) + len(narr)
    malformadas = [c[3] for c in paren if (c[2] and not c[1])]  # tiene página pero no año
    # El set de autores para el cruce sale de las citas ENTRE PARÉNTESIS (las narrativas dan
    # falsos positivos del tipo "Transdisciplinariedad (1996)"). Honesto y más robusto.
    autores_citados = set(c[0] for c in paren if c[0])

    # Lista de referencias: parsea el bloque tras el encabezado.
    refs_autores, n_refs = set(), 0
    for linea in re.split(r"\n+", refs_bloque):
        linea = linea.strip()
        if len(linea) > 12 and re.match(r"[A-ZÁÉÍÓÚÑ]", linea):
            n_refs += 1
            refs_autores.add(_autor_clave(linea))
    refs_autores.discard(None)

    huerfanas = sorted(a for a in autores_citados if refs_autores and a not in refs_autores)
    no_citadas = sorted(a for a in refs_autores if a not in autores_citados) if refs_autores else []
    segunda_mano = len(RE_SEGUNDA.findall(texto))

    return {
        "n_citas_en_texto": n_citas,
        "n_citas_parentesis": len(paren),
        "n_citas_narrativas": len(narr),
        # Swales: integral = autor como sujeto gramatical ("Morin sostiene…") ≈ narrativa;
        # no integral = entre paréntesis. Mucho no-integral sin reelaboración = "muro de citas".
        "citas_integrales": len(narr),
        "citas_no_integrales": len(paren),
        "ratio_integral": round(len(narr) / n_citas, 2) if n_citas else 0,
        "densidad_por_1000_palabras": round(1000 * n_citas / n_pal, 2) if n_pal else 0,
        "autores_distintos_citados": len(autores_citados),
        "citas_sin_anio": len(malformadas),
        "ejemplos_sin_anio": malformadas[:8],
        "n_referencias_lista": n_refs,
        "citas_huerfanas": huerfanas[:25],       # citadas en el cuerpo, ausentes en la lista
        "referencias_no_citadas": no_citadas[:25],  # en la lista, nunca citadas
        "citas_segunda_mano": segunda_mano,
        "nota": "Heurístico: matching por primer apellido sin acentos. Verificar casos límite.",
    }


# ---------- Codificación emergente (grounded) ----------
def codificacion_emergente(docs_parrafos, min_cluster=4):
    """docs_parrafos: lista de (doc_id, texto_parrafo). Devuelve códigos emergentes + cobertura."""
    textos = [p for _d, p in docs_parrafos]
    doc_ids = [d for d, _p in docs_parrafos]
    if len(textos) < 6:
        return {"metodo": "insuficiente", "nota": "Muy pocos párrafos para clustering robusto.",
                "codigos": []}

    # Vectorización
    if _EMB is not None:
        X = _EMB.encode(textos, normalize_embeddings=True, show_progress_bar=False)
        metodo_emb = "embeddings multilingües"
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(max_features=2000, stop_words=list(STOP_ES)).fit_transform(textos).toarray()
        metodo_emb = "TF-IDF"

    labels = None
    metodo_clu = None
    try:
        import hdbscan
        mcs = max(3, min(min_cluster, len(textos) // 3))
        labels = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(X)
        if len(set(labels) - {-1}) >= 2:
            metodo_clu = f"HDBSCAN (min_cluster_size={mcs})"
        else:
            labels = None
    except Exception:
        labels = None
    if labels is None:
        from sklearn.cluster import KMeans
        k = max(2, min(8, int(math.sqrt(len(textos) / 2))))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        metodo_clu = f"KMeans (k={k}) [fallback]"

    # Etiquetas TF-IDF por cluster + cita representativa + cobertura por doc
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=3000, stop_words=list(STOP_ES))
    tfidf = vec.fit_transform(textos)
    vocab = np.array(vec.get_feature_names_out())

    codigos = []
    cobertura = defaultdict(lambda: defaultdict(int))
    Xarr = np.asarray(X, dtype=float)
    for cl in sorted(set(labels) - {-1}):
        idx = [i for i, l in enumerate(labels) if l == cl]
        if len(idx) < 2:
            continue
        sub = np.asarray(tfidf[idx].mean(axis=0)).ravel()
        top = vocab[sub.argsort()[::-1][:6]]
        centroide = Xarr[idx].mean(axis=0)
        rep = idx[int(np.argmin([np.linalg.norm(Xarr[i] - centroide) for i in idx]))]
        docs_en_codigo = sorted(set(doc_ids[i] for i in idx))
        for i in idx:
            cobertura[doc_ids[i]][int(cl)] += 1
        codigos.append({
            "codigo_id": int(cl),
            "etiqueta_terminos": list(top),
            "n_segmentos": len(idx),
            "n_documentos": len(docs_en_codigo),
            "documentos": docs_en_codigo,
            "cita_representativa": textos[rep][:400],
        })

    # Co-ocurrencia entre códigos (mismos documentos)
    ids = [c["codigo_id"] for c in codigos]
    coocc = {}
    for a in ids:
        for b in ids:
            if a < b:
                docs_a = set(next(c for c in codigos if c["codigo_id"] == a)["documentos"])
                docs_b = set(next(c for c in codigos if c["codigo_id"] == b)["documentos"])
                inter = len(docs_a & docs_b)
                if inter:
                    coocc[f"{a}-{b}"] = inter

    return {
        "metodo": f"{metodo_emb} + {metodo_clu}",
        "n_codigos": len(codigos),
        "codigos": codigos,
        "co_ocurrencia": coocc,
        "cobertura_por_documento": {k: dict(v) for k, v in cobertura.items()},
    }


# ---------- Indicio LOCAL de IA (gratis, débil, SOLO interno) ----------
_FORMULAS_IA = ["cabe destacar", "es importante mencionar", "es importante señalar", "en la actualidad",
                "juega un papel", "cumple un papel", "un papel fundamental", "particularmente",
                "resulta particularmente", "asimismo", "en este sentido", "por otro lado",
                "no solo", "sino también", "cabe señalar", "es fundamental", "desempeña un papel"]


def indicio_ia_local(texto):
    """Indicio MUY DÉBIL de redacción IA, sin GPU ni costo: uniformidad de oración (burstiness/CV) +
    densidad de fórmulas de relleno. NO es prueba ni reemplaza al detector fino (Fast-DetectGPT con
    Qwen 7B en GPU). USO INTERNO del docente; jamás en el reporte del alumno ni como acusación.
    Texto IA puro tiende a burstiness muy negativo (oraciones uniformes) y muchas fórmulas; texto
    humano formal también puede puntuar — por eso es solo orientación."""
    cuerpo = cuerpo_analizable(texto)
    sents = _oraciones(cuerpo)
    L = [len(_palabras(s)) for s in sents]
    L = [x for x in L if x > 0]
    if len(L) < 5:
        return {"veredicto": "insuficiente", "n_oraciones": len(L)}
    arr = np.array(L, dtype=float)
    mean, sd = float(arr.mean()), float(arr.std())
    burstiness = round((sd - mean) / (sd + mean + 1e-9), 2)   # ~ -1 uniforme, >0 ráfagas
    cv = round(sd / (mean + 1e-9), 2)
    low = _strip(cuerpo)
    n_pal = max(1, len(_palabras(cuerpo)))
    formulas = {f: low.count(_strip(f)) for f in _FORMULAS_IA if low.count(_strip(f))}
    densidad_formulas = round(1000 * sum(formulas.values()) / n_pal, 1)
    # Heurística de banda (orientativa): IA-puro suele burstiness< -0.5 + muchas fórmulas.
    señales = (burstiness < -0.5) + (densidad_formulas > 8) + (cv < 0.30)
    banda = "alto (releer)" if señales >= 2 else ("medio (ambiguo)" if señales == 1 else "bajo")
    return {
        "veredicto": banda,
        "n_oraciones": len(L), "media_palabras": round(mean, 1), "sd": round(sd, 1),
        "burstiness": burstiness, "coef_variacion": cv,
        "densidad_formulas_x1000": densidad_formulas, "formulas": formulas,
        "nota": "Indicio débil, NO prueba. Diversidad léxica alta lo contradice. Para certeza usar "
                "el detector fino (Qwen 7B en GPU). Uso interno; nunca en el reporte del alumno.",
    }


# ---------- Red de conceptos (co-ocurrencia) ----------
def _slug(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()[:60]


def red_conceptos(texto, out_png, out_html=None, top_n=30):
    """Red de co-ocurrencia de conceptos (nodos=conceptos/entidades, aristas=co-aparición en la misma
    oración). Resalta entidades (autores/PROPN) en dorado. Devuelve stats (centralidad, si los autores
    dialogan entre sí). Requiere spaCy; si no hay, devuelve None y lo declara."""
    if _NLP is None:
        return {"metodo": "no disponible (sin spaCy)", "nodos": 0}
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doc = _NLP(texto[:1_000_000])
    # Frecuencia de conceptos (lemas de sustantivos/nombres propios). Una entidad real (autor/sigla)
    # se distingue porque va capitalizada A MEDIA oración, no solo al inicio (spaCy marca PROPN de más
    # en arranques de oración: "Sujeto…", "Formación…").
    freq, cap_mid, tot = Counter(), Counter(), Counter()
    sent_concepts = []
    for sent in doc.sents:
        presentes = set()
        for t in sent:
            if t.is_alpha and not t.is_stop and len(t) > 3 and t.pos_ in ("NOUN", "PROPN"):
                lema = t.lemma_.lower()
                if lema in STOP_ES:
                    continue
                freq[lema] += 1
                tot[lema] += 1
                if (not t.is_sent_start) and t.text[:1].isupper():
                    cap_mid[lema] += 1
                presentes.add(lema)
        if presentes:
            sent_concepts.append(presentes)
    es_entidad = {w: (tot[w] > 0 and cap_mid[w] / tot[w] >= 0.5) for w in freq}

    top = [w for w, _ in freq.most_common(top_n)]
    topset = set(top)
    pares = Counter()
    for presentes in sent_concepts:
        nodos = sorted(presentes & topset)
        for i in range(len(nodos)):
            for j in range(i + 1, len(nodos)):
                pares[(nodos[i], nodos[j])] += 1

    G = nx.Graph()
    for w in top:
        G.add_node(w, freq=freq[w], entidad=es_entidad.get(w, False))
    umbral = 2 if sum(v >= 2 for v in pares.values()) >= 8 else 1
    for (a, b), w in pares.items():
        if w >= umbral:
            G.add_edge(a, b, weight=w)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])

    if G.number_of_nodes() == 0:
        return {"metodo": "red vacía (texto muy corto)", "nodos": 0}

    # Stats básicos
    import numpy as _np
    cent = nx.degree_centrality(G)
    centrales = sorted(cent.items(), key=lambda x: -x[1])[:6]
    entidades = [n for n in G.nodes if G.nodes[n].get("entidad")]
    aristas_entre_entidades = [(a, b, G[a][b]["weight"]) for a, b in G.edges
                               if G.nodes[a].get("entidad") and G.nodes[b].get("entidad")]

    # --- Métricas de estructura (Pack A): integración vs silos ---
    estructura = {}
    try:
        from networkx.algorithms.community import louvain_communities, modularity
        coms = louvain_communities(G, weight="weight", seed=42)
        node2com = {n: i for i, c in enumerate(coms) for n in c}
        Q = modularity(G, coms, weight="weight")
        bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
        p75 = float(_np.percentile(list(bc.values()), 75)) if bc else 0.0

        def _cruza(n):  # nº de comunidades que toca el nodo (él + sus vecinos)
            return len({node2com[v] for v in G.neighbors(n)} | {node2com[n]})

        puentes = [n for n, b in bc.items() if b >= p75 and _cruza(n) >= 2]
        puentes_concepto = [n for n in puentes if not G.nodes[n].get("entidad")]
        # Índice de Puente Transdisciplinar: conceptos (no entidades) que ATRAVIESAN comunidades,
        # normalizado por nº de comunidades. Alto = hay "terceros incluidos" que religan; bajo = silos.
        ipt = round(len(puentes_concepto) / len(coms), 2) if coms else 0.0
        # Rol de cada autor/entidad: ¿hub aislado (grado alto, betweenness baja) o puente?
        roles = {}
        for n in entidades:
            roles[n] = {"grado": int(G.degree(n)), "betweenness": round(bc.get(n, 0), 3),
                        "comunidades_que_toca": _cruza(n),
                        "rol": "puente" if (bc.get(n, 0) >= p75 and _cruza(n) >= 2) else "hub local"}
        estructura = {
            "modularidad": round(Q, 3),
            "n_comunidades": len(coms),
            "frac_comunidad_mayor": round(max(len(c) for c in coms) / G.number_of_nodes(), 2),
            "indice_puente_transdisciplinar": ipt,
            "conceptos_puente": puentes_concepto[:8],
            "roles_entidades": roles,
        }
    except Exception as e:
        estructura = {"error": f"{type(e).__name__}: {e}"}

    # Dibujo estático
    import numpy as _np
    pos = nx.spring_layout(G, k=0.9, seed=42, iterations=120)
    fig, ax = plt.subplots(figsize=(11, 8))
    pesos = [G[a][b]["weight"] for a, b in G.edges]
    nx.draw_networkx_edges(G, pos, width=[0.6 + 0.9 * w for w in pesos], edge_color="#c2ccd4",
                           alpha=.7, ax=ax)
    tam = [180 + 90 * G.nodes[n]["freq"] for n in G.nodes]
    col = ["#D98C00" if G.nodes[n].get("entidad") else "#2C6E7F" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=tam, node_color=col, alpha=.92,
                           edgecolors="white", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9.5,
                            font_color="#1d2733", font_family="sans-serif", ax=ax)
    ax.set_title("Mapa de conceptos del ensayo\n"
                 "dorado = autores · tamaño = cuánto aparece · líneas = ideas que van juntas",
                 fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Versión interactiva (pyvis) opcional
    if out_html:
        try:
            from pyvis.network import Network
            net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#1d2733",
                          notebook=False, cdn_resources="remote")
            for n in G.nodes:
                ent = G.nodes[n].get("entidad")
                net.add_node(n, label=n, value=G.nodes[n]["freq"],
                             color="#D98C00" if ent else "#2C6E7F",
                             title=f"{n} · frecuencia {G.nodes[n]['freq']}")
            for a, b in G.edges:
                net.add_edge(a, b, value=G[a][b]["weight"])
            net.barnes_hut(gravity=-9000, spring_length=120)
            net.save_graph(out_html)
        except Exception as e:
            es_entidad["__pyvis_error__"] = str(e)

    return {
        "metodo": "spaCy NOUN/PROPN + co-ocurrencia por oración",
        "nodos": G.number_of_nodes(), "aristas": G.number_of_edges(),
        "umbral_arista": umbral,
        "conceptos_centrales": [{"concepto": c, "centralidad": round(v, 3)} for c, v in centrales],
        "entidades": entidades,
        "aristas_entre_entidades": [{"a": a, "b": b, "peso": w}
                                    for a, b, w in aristas_entre_entidades],
        "estructura": estructura,
        "png": os.path.basename(out_png),
        "html": os.path.basename(out_html) if out_html else None,
    }


# ---------- Estadística del grupo (lote) ----------
def estadistica_grupo(filas):
    """filas: lista de dicts planos con métricas numéricas por documento."""
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame(filas)
    n = len(df)
    cols = [c for c in df.columns if c not in ("documento", "alumno")
            and pd.api.types.is_numeric_dtype(df[c])]
    resumen, por_doc = {}, defaultdict(dict)
    for c in cols:
        serie = df[c].dropna()
        if len(serie) < 2:
            continue
        media = float(serie.mean())
        ci = None
        if len(serie) >= 3:
            try:
                res = stats.bootstrap((serie.values,), np.mean, n_resamples=2000,
                                      confidence_level=0.95, method="percentile")
                ci = [round(float(res.confidence_interval.low), 3),
                      round(float(res.confidence_interval.high), 3)]
            except Exception:
                ci = None
        resumen[c] = {"media": round(media, 3), "sd": round(float(serie.std()), 3),
                      "min": round(float(serie.min()), 3), "max": round(float(serie.max()), 3),
                      "ic95_media": ci}
        sd = serie.std()
        for _, row in df.iterrows():
            val = row[c]
            if pd.isna(val):
                continue
            doc = row["documento"]
            por_doc[doc][c] = {
                "valor": round(float(val), 3),
                "percentil": round(float(stats.percentileofscore(serie, val)), 1),
                "z": round(float((val - media) / sd), 2) if sd else 0.0,
            }
    return {
        "n_documentos": n,
        "muestra_pequena": n < 8,
        "advertencia": ("Con n<8 los percentiles y z-scores son orientativos, no inferenciales."
                        if n < 8 else None),
        "resumen_grupo": resumen,
        "por_documento": {k: dict(v) for k, v in por_doc.items()},
    }


# ---------- Orquestación ----------
def analizar_documento(texto):
    cuerpo = cuerpo_analizable(texto)   # sin portada ni referencias
    return {
        # Conteo CANÓNICO: 'cuerpo' = palabras de CONTENIDO (sin portada/nombre/datos ni lista de
        # referencias). Es el ÚNICO número que se cita en los reportes. 'total_crudo' queda solo de
        # referencia interna para no confundirlo. Regla de Alex: importa el contenido, no la portada.
        "conteo": {
            "palabras_cuerpo": len(_palabras(cuerpo)),
            "palabras_total_crudo": len(_palabras(texto)),
        },
        "lexico": metricas_lexicas(cuerpo),
        "legibilidad": metricas_legibilidad(cuerpo),
        "sintaxis": metricas_sintaxis(cuerpo),          # Pack B
        "cohesion": metricas_cohesion(cuerpo),
        "argumentacion": metricas_argumentacion(cuerpo),  # Pack A
        "citas": metricas_citas(texto),   # citas usa el texto completo (necesita la lista de refs)
    }


def _fila_plana(stem, alumno, m):
    s, a = m.get("sintaxis", {}), m.get("argumentacion", {})
    return {
        "documento": stem, "alumno": alumno,
        "palabras": m["lexico"]["n_tokens"],
        "ttr": m["lexico"]["ttr"], "guiraud": m["lexico"]["guiraud"], "mtld": m["lexico"]["mtld"],
        "mattr": m["lexico"].get("mattr"),
        "densidad_lexica": m["lexico"].get("densidad_lexica"),
        "fernandez_huerta": m["legibilidad"]["fernandez_huerta"],
        "palabras_por_oracion": m["legibilidad"]["palabras_por_oracion"],
        "oraciones_largas_pct": m["legibilidad"]["oraciones_largas_pct"],
        "mdd": s.get("mdd"), "indice_subordinacion": s.get("indice_subordinacion"),
        "profundidad_arbol": s.get("profundidad_arbol_media"),
        "ratio_nominalizacion": s.get("ratio_nominalizacion"),
        "densidad_proposicional": s.get("densidad_proposicional"),
        "conectores_x100": m["cohesion"]["conectores_por_100_oraciones"],
        "cohesion_lexica": m["cohesion"].get("cohesion_lexica_parrafos"),
        "coherencia_semantica": m["cohesion"].get("coherencia_semantica"),
        "coherencia_global": m["cohesion"].get("coherencia_global"),
        "ratio_global_local": m["cohesion"].get("ratio_global_local"),
        "pct_respaldo": a.get("pct_oraciones_con_respaldo"),
        "densidad_causal_x100": a.get("densidad_causal_x100"),
        "citas_total": m["citas"]["n_citas_en_texto"],
        "citas_densidad_1000": m["citas"]["densidad_por_1000_palabras"],
        "ratio_integral": m["citas"].get("ratio_integral"),
        "citas_sin_anio": m["citas"]["citas_sin_anio"],
        "citas_huerfanas": len(m["citas"]["citas_huerfanas"]),
    }


def run(revision_dir):
    """Lee manifest.json, calcula todo, escribe artefactos. Devuelve un dict-resumen para el notebook."""
    man_path = os.path.join(revision_dir, "manifest.json")
    manifest = json.load(open(man_path, encoding="utf-8"))
    os.makedirs(os.path.join(revision_dir, "metricas"), exist_ok=True)
    os.makedirs(os.path.join(revision_dir, "figuras"), exist_ok=True)
    os.makedirs(os.path.join(revision_dir, "redes"), exist_ok=True)

    filas, docs_parrafos, redes = [], [], {}
    for e in manifest["ensayos"]:
        texto = open(e["texto_path"], encoding="utf-8").read()
        m = analizar_documento(texto)
        with open(os.path.join(revision_dir, "metricas", e["stem"] + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump({"alumno": e["alumno"], "archivo": e["archivo_original"], **m},
                      f, ensure_ascii=False, indent=2)
        filas.append(_fila_plana(e["stem"], e["alumno"], m))
        cuerpo = cuerpo_analizable(texto)   # sin portada ni referencias
        for p in _parrafos(cuerpo):
            docs_parrafos.append((e["stem"], p))
        # Red de conceptos por documento (sobre el cuerpo limpio)
        slug = _slug(e["stem"])
        png = os.path.join(revision_dir, "figuras", f"red_conceptos_{slug}.png")
        html = os.path.join(revision_dir, "redes", f"red_{slug}.html")
        try:
            redes[e["stem"]] = red_conceptos(cuerpo, png, html)
        except Exception as ex:
            redes[e["stem"]] = {"metodo": f"error: {ex}", "nodos": 0}

    # Codificación emergente (dentro de un doc si n=1; cross-doc si lote)
    cod = codificacion_emergente(docs_parrafos)
    json.dump(cod, open(os.path.join(revision_dir, "codificacion.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Estadística del grupo solo si ≥2 documentos
    cohorte = None
    if len(filas) >= 2:
        cohorte = estadistica_grupo(filas)
        json.dump(cohorte, open(os.path.join(revision_dir, "cohorte.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    json.dump(filas, open(os.path.join(revision_dir, "tabla_metricas.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(redes, open(os.path.join(revision_dir, "redes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    return {
        "n_documentos": len(filas),
        "componentes": {"spacy": _NLP is not None, "embeddings": _EMB is not None},
        "codificacion": {"metodo": cod.get("metodo"), "n_codigos": cod.get("n_codigos", 0)},
        "redes": {k: {"nodos": v.get("nodos"), "aristas_entre_entidades":
                      len(v.get("aristas_entre_entidades", []))} for k, v in redes.items()},
        "tiene_cohorte": cohorte is not None,
        "tabla": filas,
    }
