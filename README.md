# Motor de minería de texto para la evaluación de la competencia investigativa

Código del motor de minería descrito en el capítulo *Evaluación asistida por
inteligencia artificial de la competencia investigativa en posgrado: minería de
texto y verificación de citas* (Duarte Velázquez, en dictamen; libro UPN Unidad
161 Morelia / UVAQ / UNIVIM sobre Evaluación en la educación superior).

## Qué hace

Sobre un texto académico en español calcula, de forma reproducible:

- **Diversidad léxica**: TTR, índice de Guiraud (diagnóstico interno), MTLD
  bidireccional (umbral 0.72) y MATTR (ventana de 50 palabras).
- **Legibilidad para el español**: Fernández-Huerta y Szigriszt-Pazos (escala INFLESZ).
- **Cohesión y coherencia**: densidad de conectores por cada cien oraciones,
  solapamiento léxico entre párrafos, coherencia semántica por embeddings de
  oración (paraphrase-multilingual-MiniLM-L12-v2) y saltos temáticos.
- **Aparato crítico**: detección de citas parentéticas y narrativas, validación de
  formato APA 7, cruce cita-referencia bidireccional y marcadores de citas de
  segunda mano.
- **Codificación emergente**: agrupamiento semántico de párrafos (HDBSCAN, con
  degradación declarada a KMeans), términos por TF-IDF y cita ancla por grupo.

Principio de diseño: **el motor mide; el revisor juzga**. Toda degradación de
método se declara en el propio resultado, los cruces heurísticos exigen
verificación humana y ninguna métrica sustituye la lectura experta.

## Uso

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_md   # opcional
python correr_mineria.py <carpeta_con_txt>
```

Si `spacy`, `sentence-transformers` o `hdbscan` no están disponibles, el motor
recurre a métodos alternativos y lo deja registrado en el campo de método del
resultado (no se corrige a mano).

## Licencia

MIT. Si usas este código en un trabajo académico, cita el capítulo.
