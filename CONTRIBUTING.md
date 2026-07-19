# Contribuir a aprende-vslam

*(English summary at the end.)*

Gracias por tu interés. Este repo es un curso: la vara para aceptar cambios
no es solo "¿funciona?", sino "¿enseña mejor?". Estas guías explican cómo
reportar problemas, proponer mejoras y mandar PRs que pasen la revisión.

## Reportar un problema

Abre un [issue](https://github.com/ariel9874/aprende-vslam/issues) con:

1. **El nivel** y el script exacto (p. ej. `nivel_11_bundle_adjustment/verificacion.py`).
2. **La salida completa** del error (los exámenes imprimen números medidos:
   inclúyelos — son el diagnóstico).
3. Tu sistema operativo y versión de Python (`python --version`).
4. Si el nivel usa dataset: cuál y si lo descargó el script o lo pasaste
   con `--root`.

¿Duda conceptual en lugar de bug? También es bienvenida como issue — las
preguntas frecuentes se convierten en mejoras del README del nivel.

## La constitución (lo que un PR NO puede romper)

Las seis reglas del [README](README.md#las-reglas-del-curso-la-constitución)
son innegociables; las tres que más PRs rechazan:

- **Independencia total**: cero imports entre niveles, cero paquete común.
  Si tu mejora sirve a dos niveles, se DUPLICA en ambos (sí, en serio: el
  contexto inmediato prima sobre el DRY — es un curso, no una librería).
- **Cada afirmación, medida**: si tocas un algoritmo, el examen del nivel
  debe seguir pasando; si cambias un comportamiento medido, actualiza el
  número esperado EN el examen, el README del nivel y la tabla del README
  raíz, y explica en el PR de dónde sale el número nuevo.
- **Windows-proof**: prints en ASCII (los docstrings y READMEs sí llevan
  acentos), rutas con `pathlib`, nada que dependa de un shell concreto.

## Antes de mandar el PR

```bash
cd nivel_XX_lo_que_tocaste
python verificacion.py          # el examen del nivel: debe dar VERIFICADO

python verifica_todos.py        # (mantenedores) todos los niveles;
                                # acepta --root para reusar datasets
```

El CI corre los exámenes sin dataset en Ubuntu y Windows: tu PR debe salir
en verde. Si tu cambio necesita un dataset para verificarse, dilo en la
descripción del PR y pega la salida local del examen.

## Proponer un nivel o un ejercicio nuevo

Un nivel nuevo se discute primero en un issue. El molde es fijo: scripts
planos legibles de arriba a abajo + driver + `verificacion.py` con números
medidos + `README.md` con la intuición + `EJERCICIOS.md` (cada ejercicio
con su número objetivo) + `requirements.txt`. La pregunta que decide si un
nivel entra: **¿qué número medible aprende el alumno a reproducir?**

Los ejercicios nuevos para niveles existentes son la contribución más fácil
y más valiosa: mandalos directo como PR a `EJERCICIOS.md`.

## Código de conducta

Este proyecto se rige por el [Código de Conducta](CODE_OF_CONDUCT.md)
(Contributor Covenant). Al participar, lo aceptas.

---

## English summary

This is a Spanish-language, exam-driven VSLAM course. Contributions are
welcome (issues and PRs in English are fine). Hard rules: levels are fully
self-contained (no cross-level imports — duplication is deliberate), every
claim is backed by a measured number in the level's `verificacion.py` exam
(CI runs the dataset-free exams on Ubuntu and Windows), and scripts must be
Windows-proof (ASCII prints, `pathlib`). New levels start as an issue; new
exercises for existing levels can go straight to a PR. The question that
decides whether a change is accepted: *does it teach better, and is the
number still measured?*
